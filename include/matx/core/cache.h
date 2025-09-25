////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#ifndef __CUDACC_RTC__
#pragma once

#include <functional>
#include <optional>
#include <any>
#include <shared_mutex>
#include <unordered_map>
#include <cuda/atomic>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <cstring>

#include "matx/core/error.h"

namespace matx {
namespace detail {

/**
 * @brief Structure to hold LTOIR byte data with size information
 */
struct LTOIRData {
  char* data = nullptr;      // Raw pointer to byte data  
  size_t length = 0;         // Size in bytes

  // Destructor to free allocated memory
  ~LTOIRData() {
    free(data);  // free(nullptr) is safe
  }

  LTOIRData() = default;
  LTOIRData(char* d, size_t l) : data(d), length(l) {}
    
  // Move constructor
  LTOIRData(LTOIRData&& other) noexcept : data(other.data), length(other.length) {
    other.data = nullptr;
    other.length = 0;
  }
  
  // Move assignment
  LTOIRData& operator=(LTOIRData&& other) noexcept {
    if (this != &other) {
      free(data);
      data = other.data;
      length = other.length;
      other.data = nullptr;
      other.length = 0;
    }
    return *this;
  }
  
  // Delete copy constructor and assignment
  LTOIRData(const LTOIRData&) = delete;
  LTOIRData& operator=(const LTOIRData&) = delete;
};

static constexpr size_t MAX_CUDA_DEVICES_PER_SYSTEM = 16;
using CacheId = uint64_t;

#ifndef DOXYGEN_ONLY
__attribute__ ((visibility ("default")))
#endif
inline cuda::std::atomic<CacheId> CacheIdCounter{0};
inline std::recursive_mutex cache_mtx; ///< Mutex protecting updates from map

template<typename CacheType>
__attribute__ ((visibility ("default")))
CacheId GetCacheIdFromType()
{
  static CacheId id = CacheIdCounter.fetch_add(1);

  return id;
}

struct StreamAllocation {
  void* ptr;
  size_t size;
};

/**
 * Generic caching object for caching parameters. This class is used for
 * creating handles/plans on-the-fly and caching them to remove the need for
 * plans on certain interfaces. For example, InParams can be all parameters
 * needed to define an FFT, and if that plan already exists, a user doesn't need
 * to create another plan.
 */
class matxCache_t {
public:
  matxCache_t() {}
  ~matxCache_t() {
    // Destroy all outstanding objects in the cache to free memory
    for (auto &[k, v]: cache) {
      v.reset();
    }
  }

  /**
   * Deletes the entire contents of the cache
   *
   */
  template <typename CacheType>
  void Clear(const CacheId &id) {
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(cache_mtx);

    auto el = cache.find(id);
    MATX_ASSERT_STR(el != cache.end(), matxInvalidType, "Cache type not found");

    for (int i = 0; i < static_cast<int>(MAX_CUDA_DEVICES_PER_SYSTEM); i++) {
      using CacheArray = cuda::std::array<CacheType, MAX_CUDA_DEVICES_PER_SYSTEM>;
      std::any_cast<CacheArray&>(el->second)[i].clear();
    }
  }

  template <typename CacheType, typename InParams, typename MakeFun, typename ExecFun, typename Executor>
  void LookupAndExec(const CacheId &id, const InParams &params, const MakeFun &mfun, const ExecFun &efun, [[maybe_unused]] const Executor &exec) {
    // This mutex should eventually be finer-grained so each transform doesn't get blocked by others
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(cache_mtx);
    using CacheArray = cuda::std::array<CacheType, MAX_CUDA_DEVICES_PER_SYSTEM>;

    // Create named cache if it doesn't exist
    int device_id;
    auto el = cache.find(id);
    if (el == cache.end()) {
      cache[id] = CacheArray{};
    }

    auto &cval = cache[id];
    if constexpr (is_cuda_executor_v<Executor>) {
      cudaGetDevice(&device_id);
    }
    else {
      device_id = 0;
    }

    auto &rmap = std::any_cast<CacheArray&>(cval)[device_id];
    auto cache_el = rmap.find(params);
    if (cache_el == rmap.end()) {
      std::any tmp = mfun();
      rmap.insert({params, tmp});
      efun(std::any_cast<decltype(mfun())>(tmp));
    }
    else {
      efun(std::any_cast<decltype(mfun())>(cache_el->second));
    }
  }

  void* GetStreamAlloc(cudaStream_t stream, size_t size) {
    void *ptr = nullptr;
    int device_id;
    cudaGetDevice(&device_id);

    auto el = stream_alloc_cache[device_id].find(stream);
    if (el == stream_alloc_cache[device_id].end()) {
      StreamAllocation alloc;

      // We allocate at least 2MB for workspace so we don't keep reallocating from small sizes
      size = std::max(size, (size_t)(1ULL << 21));
      matxAlloc(&ptr, size, MATX_ASYNC_DEVICE_MEMORY, stream);

      alloc.size = size;
      alloc.ptr = ptr;
      stream_alloc_cache[device_id][stream] = alloc;
    }
    else if (el->second.size < size) {
      // Free the old allocation and allocate a new one
      matxFree(el->second.ptr);
      matxAlloc(&ptr, size, MATX_ASYNC_DEVICE_MEMORY, stream);
      el->second.size = size;
      el->second.ptr = ptr;
    }
    else {
      ptr = el->second.ptr;
    }

    return ptr;
  }

  /**
   * @brief Helper function to determine the cache directory path
   * 
   * @return std::string The cache directory path, or empty string if unavailable
   */
  __MATX_INLINE__ std::string GetKernelCacheDirectory() {
#ifndef __CUDACC_RTC__
    std::string cache_dir;
    const char* env_cache_dir = std::getenv("MATX_CACHE_DIR");
    if (env_cache_dir) {
      cache_dir = env_cache_dir;
    } else {
      const char* home = std::getenv("HOME");
      if (home) {
        cache_dir = std::string(home) + "/.matx/kernel_cache";
        // Create the directory if it doesn't exist
        std::filesystem::create_directories(cache_dir);
      }
    }
    return cache_dir;
#else
    return "";
#endif
  }

  

  /**
  * @brief Look up cached data by filename
  * 
  * This function checks if a given filename exists in a two-level cache:
  * 1. First checks an in-memory unordered_map cache
  * 2. If not found, searches the filesystem cache directory
  * 
  * The cache directory is determined by:
  * - Environment variable MATX_CACHE_DIR if it exists
  * - ${HOME}/.matx/kernel_cache otherwise
  * 
  * @param filename The name of the file to look up
  * @return LTOIRData* Pointer to cached data structure if found, nullptr otherwise
  */
  __MATX_INLINE__ LTOIRData* GetLTOIRCachedBytes(const std::string& filename) {
#ifndef __CUDACC_RTC__
    // First check the in-memory cache
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      return &it->second;
    }
    
    // Determine cache directory
    std::string cache_dir = GetKernelCacheDirectory();
    if (cache_dir.empty()) {
      return nullptr; // No cache directory available
    }
    
    // Check if file exists in cache directory
    std::filesystem::path cache_file = std::filesystem::path(cache_dir) / filename;
    if (!std::filesystem::exists(cache_file)) {
      return nullptr;
    }
    
    // Read file contents
    std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return nullptr;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char* buffer = static_cast<char*>(malloc(size));
    if (!buffer) {
      return nullptr;
    }
    
    if (!file.read(buffer, size)) {
      free(buffer);
      return nullptr;
    }
    
    ltoir_cache[filename] = LTOIRData{buffer, static_cast<size_t>(size)};
    
    return &ltoir_cache[filename];
  #else
    // In RTC mode, return nullptr as filesystem operations are not available
    return nullptr;
  #endif
  }

  /**
   * @brief Store byte array data in the LTO IR cache by transferring ownership
   * 
   * This function stores the provided raw pointer directly in the cache by transferring
   * ownership to the cache. The cache will manage the memory with free().
   * Also stores the data to disk in the same location that GetLTOIRCachedBytes reads from.
   * 
   * @param filename The key to store the data under
   * @param data Raw pointer to the byte array (ownership transferred to cache)
   * @param length Size of the data in bytes
   * @return true if successfully stored, false otherwise
   */
  __MATX_INLINE__ bool StoreLTOIRCachedBytes(const std::string& filename, char* data, size_t length) {
#ifndef __CUDACC_RTC__
    if (!data || length == 0) {
      return false;
    }
    
    try {
      // Store the raw pointer and length in an LTOIRData struct
      ltoir_cache[filename] = LTOIRData{data, length};
      
      // Also store to disk for persistence
      std::string cache_dir = GetKernelCacheDirectory();
      if (!cache_dir.empty()) {
        try {
          // Create cache directory if it doesn't exist
          std::filesystem::create_directories(cache_dir);
          
          // Write to disk
          std::filesystem::path cache_file = std::filesystem::path(cache_dir) / filename;
          std::ofstream file(cache_file, std::ios::binary);
          if (file.is_open()) {
            file.write(data, length);
            file.close();
          }
          // Note: We don't fail if disk write fails, as in-memory cache is still valid
        } catch (...) {
          // Ignore disk write failures - in-memory cache is still valid
        }
      }
      
      return true;
    } catch (...) {
      // Handle any failures - free the data since we couldn't store it
      free(data);
      return false;
    }
#else
    // In RTC mode, caching is not available
    free(data);
    return false;
#endif
  }

  /**
   * @brief Store byte array data in the LTO IR cache by copying
   * 
   * This function stores a copy of the provided byte array in the in-memory cache
   * using the specified filename as the key.
   * Also stores the data to disk in the same location that GetLTOIRCachedBytes reads from.
   * 
   * @param filename The key to store the data under
   * @param data Pointer to the byte data to store
   * @param size Size of the data in bytes
   * @return true if successfully stored, false otherwise
   */
  __MATX_INLINE__ bool StoreLTOIRCachedBytes(const std::string& filename, const char* data, size_t size) {
#ifndef __CUDACC_RTC__
    if (!data || size == 0) {
      return false;
    }
    
    try {
      // Create a new buffer and copy the data
      char* buffer = static_cast<char*>(malloc(size));
      if (!buffer) {
        return false;
      }
      std::memcpy(buffer, data, size);
      
      // Store in the cache with size information
      ltoir_cache[filename] = LTOIRData{buffer, size};
      
      // Also store to disk for persistence
      std::string cache_dir = GetKernelCacheDirectory();
      if (!cache_dir.empty()) {
        try {
          // Create cache directory if it doesn't exist
          std::filesystem::create_directories(cache_dir);
          
          // Write to disk
          std::filesystem::path cache_file = std::filesystem::path(cache_dir) / filename;
          std::ofstream file(cache_file, std::ios::binary);
          if (file.is_open()) {
            file.write(data, size);
            file.close();
          }
          // Note: We don't fail if disk write fails, as in-memory cache is still valid
        } catch (...) {
          // Ignore disk write failures - in-memory cache is still valid
        }
      }
      
      return true;
    } catch (...) {
      // Handle any allocation or copy failures
      return false;
    }
#else
    // In RTC mode, caching is not available
    return false;
#endif
  }

  /**
   * @brief Get the size of cached data
   * 
   * @param filename The key to check for
   * @return Size of the cached data in bytes, or 0 if not found
   */
  __MATX_INLINE__ size_t GetLTOIRCachedBytesLength(const std::string& filename) {
#ifndef __CUDACC_RTC__
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      return it->second.length;
    }
    return 0;
#else
    return 0;
#endif
  }

  /**
   * @brief Check if a key exists in the LTO IR cache
   * 
   * @param filename The key to check for
   * @return true if the key exists in the cache, false otherwise
   */
  __MATX_INLINE__ bool HasLTOIRCachedBytes(const std::string& filename) {
#ifndef __CUDACC_RTC__
    return ltoir_cache.find(filename) != ltoir_cache.end();
#else
    return false;
#endif
  }

  /**
   * @brief Remove an entry from the LTO IR cache
   * 
   * @param filename The key to remove
   * @return true if the key was found and removed, false if it didn't exist
   */
  __MATX_INLINE__ bool RemoveLTOIRCachedBytes(const std::string& filename) {
#ifndef __CUDACC_RTC__
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      ltoir_cache.erase(it);
      return true;
    }
    return false;
#else
    return false;
#endif
  }


private:
  // Static cache for in-memory storage
  std::unordered_map<std::string, LTOIRData> ltoir_cache;
  std::unordered_map<CacheId, std::any> cache;
  cuda::std::array<std::unordered_map<cudaStream_t, StreamAllocation>, MAX_CUDA_DEVICES_PER_SYSTEM> stream_alloc_cache;
};

/**
 * Converts elements in a POD container to a hash value
 */
template <typename T, int len>
inline size_t PodArrayToHash(cuda::std::array<T, len> c)
{
  size_t hash = 0;
  for (auto &el : c) {
    hash += std::hash<T>()(el);
  }

  return hash;
}

__attribute__ ((visibility ("default")))
__MATX_INLINE__ matxCache_t &InitCache() {
  static matxCache_t cache;
  return cache;
}

__attribute__ ((visibility ("default")))
__MATX_INLINE__ matxCache_t &GetCache() {
  [[maybe_unused]] const auto &tracker = GetAllocMap();
  return InitCache();
}




}  // namespace detail
}; // namespace matx
#endif
