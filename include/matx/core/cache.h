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
#include <thread>

#include "matx/core/error.h"
#include "matx/core/allocator.h"
#include "matx/core/type_utils_both.h"
#include "matx/core/log.h"

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

// Common cache parameters that every cache entry needs
struct CacheCommonParamsKey {
  int device_id;
  std::thread::id thread_id;
  
  bool operator==(const CacheCommonParamsKey& other) const {
    return device_id == other.device_id && thread_id == other.thread_id;
  }
};

struct CacheCommonParamsKeyHash {
  std::size_t operator()(const CacheCommonParamsKey& key) const {
    std::size_t h1 = std::hash<int>{}(key.device_id);
    std::size_t h2 = std::hash<std::thread::id>{}(key.thread_id);
    return h1 ^ (h2 << 1);
  }
};

#ifndef DOXYGEN_ONLY
__attribute__ ((visibility ("default")))
#endif
inline cuda::std::atomic<CacheId> CacheIdCounter{0};
inline std::recursive_mutex cache_mtx; ///< Mutex protecting updates from map
inline std::recursive_mutex ltoir_mutex; ///< Mutex protecting LTOIR cache operations

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

    using CacheMap = std::unordered_map<CacheCommonParamsKey, CacheType, CacheCommonParamsKeyHash>;
    std::any_cast<CacheMap&>(el->second).clear();
  }

  template <typename CacheType, typename InParams, typename MakeFun, typename ExecFun, typename Executor>
  void LookupAndExec(const CacheId &id, const InParams &params, const MakeFun &mfun, const ExecFun &efun, [[maybe_unused]] const Executor &exec) {
    // This mutex should eventually be finer-grained so each transform doesn't get blocked by others
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(cache_mtx);
    using CacheMap = std::unordered_map<CacheCommonParamsKey, CacheType, CacheCommonParamsKeyHash>;

    // Create named cache if it doesn't exist
    CacheCommonParamsKey key;
    key.thread_id = std::this_thread::get_id();
    
    auto el = cache.find(id);
    if (el == cache.end()) {
      cache[id] = CacheMap{};
    }

    auto &cval = cache[id];
    if constexpr (is_cuda_executor_v<Executor>) {
      cudaGetDevice(&key.device_id);
    }
    else {
      key.device_id = 0;
    }

    auto &rmap = std::any_cast<CacheMap&>(cval);
    auto &common_params_cache = rmap[key];
    auto cache_el = common_params_cache.find(params);
    if (cache_el == common_params_cache.end()) {
      std::any tmp = mfun();
      common_params_cache.insert({params, tmp});
      efun(std::any_cast<decltype(mfun())>(tmp));
    }
    else {
      efun(std::any_cast<decltype(mfun())>(cache_el->second));
    }
  }

  void* GetStreamAlloc(cudaStream_t stream, size_t size) {
    void *ptr = nullptr;
    CacheCommonParamsKey key;
    key.thread_id = std::this_thread::get_id();
    cudaGetDevice(&key.device_id);

    auto &common_params_cache = stream_alloc_cache[key];
    auto el = common_params_cache.find(stream);
    if (el == common_params_cache.end()) {
      StreamAllocation alloc;

      // We allocate at least 2MB for workspace so we don't keep reallocating from small sizes
      size = std::max(size, (size_t)(1ULL << 21));
      matxAlloc(&ptr, size, MATX_ASYNC_DEVICE_MEMORY, stream);

      alloc.size = size;
      alloc.ptr = ptr;
      common_params_cache[stream] = alloc;
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
   * @brief Convert a C++ type string into a valid filename
   * 
   * Takes a complex C++ type string (with templates, namespaces, etc.) and converts
   * it into a safe filename that can be used for caching cubin/LTOIR files.
   * The function creates a deterministic hash-based filename to handle arbitrary
   * type complexity while keeping filenames short and filesystem-safe.
   * 
   * @param kernel_op_type The C++ type string to convert
   * @return std::string A safe filename derived from the type string
   */
  __MATX_INLINE__ std::string TypeStringToFilename(const std::string& kernel_op_type) {
    // Compute a hash of the full type string for uniqueness
    std::hash<std::string> hasher;
    size_t hash_value = hasher(kernel_op_type);
    
    // Convert hash to hex string for filename
    char hash_str[17]; // 16 hex chars + null terminator
    snprintf(hash_str, sizeof(hash_str), "%016zx", hash_value);
    
    // Extract a prefix from the type string for readability
    // Find the first template or operator name
    std::string prefix;
    size_t first_bracket = kernel_op_type.find('<');
    if (first_bracket != std::string::npos && first_bracket > 0) {
      // Use the first operator/class name before the first '<'
      prefix = kernel_op_type.substr(0, std::min(first_bracket, size_t(32)));
    } else {
      // No templates, use first 32 chars
      prefix = kernel_op_type.substr(0, std::min(kernel_op_type.length(), size_t(32)));
    }
    
    // Clean the prefix to remove any invalid filename characters
    for (char& c : prefix) {
      if (c == '<' || c == '>' || c == ':' || c == '/' || c == '\\' || 
          c == '|' || c == '?' || c == '*' || c == '"' || c == ' ' || c == ',') {
        c = '_';
      }
    }
    
    // Combine prefix with hash for a readable yet unique filename
    return prefix + "_" + std::string(hash_str) + ".cubin";
  }

  /**
   * @brief Helper function to determine the cache directory path
   * 
   * @return std::string The cache directory path, or empty string if unavailable
   */
  __MATX_INLINE__ std::string GetKernelCacheDirectory() {
    std::string cache_dir;
    const char* env_cache_dir = std::getenv("MATX_CACHE_DIR");
    if (env_cache_dir) {
      cache_dir = env_cache_dir;
      MATX_LOG_DEBUG("Using cache directory from MATX_CACHE_DIR: {}", cache_dir);
    } else {
      const char* home = std::getenv("HOME");
      if (home) {
        cache_dir = std::string(home) + "/.matx/kernel_cache";
        // Create the directory if it doesn't exist
        try {
          std::filesystem::create_directories(cache_dir);
          MATX_LOG_DEBUG("Created cache directory: {}", cache_dir);
        } catch (const std::exception& e) {
          MATX_LOG_ERROR("Failed to create cache directory {}: {}", cache_dir, e.what());
        }
      } else {
        MATX_LOG_WARN("No HOME environment variable set, cache directory unavailable");
      }
    }
    return cache_dir;
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
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    // First check the in-memory cache
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      MATX_LOG_DEBUG("Cache HIT (memory) for: {}", filename);
      return &it->second;
    }
    
    // Determine cache directory
    std::string cache_dir = GetKernelCacheDirectory();
    if (cache_dir.empty()) {
      MATX_LOG_DEBUG("Cache MISS for {}: no cache directory available", filename);
      return nullptr; // No cache directory available
    }
    
    // Check if file exists in cache directory
    std::filesystem::path cache_file = std::filesystem::path(cache_dir) / filename;
    if (!std::filesystem::exists(cache_file)) {
      MATX_LOG_DEBUG("Cache MISS (disk) for: {}", filename);
      return nullptr;
    }
    
    // Read file contents
    std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      MATX_LOG_ERROR("Failed to open cached file: {}", cache_file.string());
      return nullptr;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    MATX_LOG_DEBUG("Cache HIT (disk) for: {}, size: {} bytes", filename, size);
    
    char* buffer = static_cast<char*>(malloc(size));
    if (!buffer) {
      MATX_LOG_ERROR("Failed to allocate {} bytes for cache file: {}", size, filename);
      return nullptr;
    }
    
    // Use RAII to ensure buffer is freed if any exception occurs before ownership transfer
    std::unique_ptr<char, decltype(&free)> buffer_guard(buffer, &free);
    
    if (!file.read(buffer, size)) {
      MATX_LOG_ERROR("Failed to read cache file: {}", filename);
      return nullptr;  // buffer_guard automatically frees buffer
    }
    
    // Basic validation: check if data looks reasonable (not all zeros, has some content)
    // Note: LTOIR format may vary (LLVM bitcode 'BC', NVVM IR, compressed, etc.)
    // so we don't validate specific magic bytes, just sanity check
    if (size >= 4) {
      // Check if it looks like garbage (all same byte pattern often indicates corruption)
      bool looks_corrupt = (buffer[0] == buffer[1] && buffer[1] == buffer[2] && 
                           buffer[2] == buffer[3] && buffer[0] == 0);
      if (looks_corrupt) {
        MATX_LOG_WARN("Cached LTOIR file '{}' appears corrupted (all zeros), removing", filename);
        // buffer_guard will automatically free buffer
        try {
          std::filesystem::remove(cache_file);
          MATX_LOG_DEBUG("Removed corrupted cache file: {}", filename);
        } catch (const std::exception& e) {
          MATX_LOG_ERROR("Failed to remove corrupted cache file {}: {}", filename, e.what());
        }
        return nullptr;
      }
    }
    
    // IMPORTANT: Reserve space to prevent rehashing which would invalidate existing pointers
    // Reserve space for at least 32 more entries to reduce rehashing probability
    // Note: reserve() may throw, but buffer_guard ensures buffer is freed
    if (ltoir_cache.size() >= ltoir_cache.bucket_count() * static_cast<size_t>(ltoir_cache.max_load_factor()) - 1) {
      ltoir_cache.reserve(ltoir_cache.size() + 32);
    }
    
    // Transfer ownership to LTOIRData (release from buffer_guard)
    ltoir_cache[filename] = LTOIRData{buffer_guard.release(), static_cast<size_t>(size)};
    
    return &ltoir_cache[filename];
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
    if (!data || length == 0) {
      MATX_LOG_ERROR("Cannot store empty data for: {}", filename);
      return false;
    }
    
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    try {
      // Use RAII to manage ownership until all operations complete successfully
      std::unique_ptr<char, decltype(&free)> data_guard(data, &free);
      
      // Write to disk first (before transferring ownership to map)
      // This way if disk write throws, data_guard will clean up automatically
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
            MATX_LOG_DEBUG("Stored {} bytes to disk cache: {}", length, cache_file.string());
          } else {
            MATX_LOG_WARN("Failed to open file for writing: {}", cache_file.string());
          }
          // Note: We don't fail if disk write fails, as in-memory cache is still valid
        } catch (const std::exception& e) {
          MATX_LOG_WARN("Failed to write to disk cache for {}: {}", filename, e.what());
          // Ignore disk write failures - in-memory cache is still valid
        }
      }
      
      // Transfer ownership to LTOIRData only after disk operations complete
      ltoir_cache[filename] = LTOIRData{data_guard.release(), length};
      MATX_LOG_DEBUG("Stored {} bytes in memory cache for: {}", length, filename);
      
      return true;
    } catch (const std::exception& e) {
      // Handle any failures - data_guard or map destructor will free the data
      MATX_LOG_ERROR("Failed to store cache data for {}: {}", filename, e.what());
      return false;
    }
  }

  /**
   * @brief Store metadata string (like lowered kernel name) for a cached cubin
   * 
   * Stores a metadata string (e.g., lowered kernel name) in a .meta file alongside
   * the cached cubin file. The metadata file has the same name as the cubin but with
   * ".meta" appended.
   * 
   * @param filename The cubin filename (e.g., "kernel_abc123.cubin")
   * @param metadata The metadata string to store (e.g., lowered kernel name)
   * @return true if successfully stored, false otherwise
   */
  __MATX_INLINE__ bool StoreLTOIRMetadata(const std::string& filename, const std::string& metadata) {
    std::string cache_dir = GetKernelCacheDirectory();
    if (cache_dir.empty()) {
      MATX_LOG_DEBUG("Cannot store metadata for {}: no cache directory", filename);
      return false;
    }
    
    try {
      std::filesystem::create_directories(cache_dir);
      std::filesystem::path meta_file = std::filesystem::path(cache_dir) / (filename + ".meta");
      std::ofstream file(meta_file);
      if (file.is_open()) {
        file << metadata;
        file.close();
        MATX_LOG_DEBUG("Stored metadata for {}: {}", filename, metadata);
        return true;
      } else {
        MATX_LOG_ERROR("Failed to open metadata file for writing: {}", meta_file.string());
      }
    } catch (const std::exception& e) {
      MATX_LOG_ERROR("Failed to store metadata for {}: {}", filename, e.what());
    }
    return false;
  }

  /**
   * @brief Retrieve metadata string for a cached cubin
   * 
   * Loads the metadata string from the .meta file associated with a cached cubin.
   * 
   * @param filename The cubin filename (e.g., "kernel_abc123.cubin")
   * @return The metadata string if found, or empty string if not found
   */
  __MATX_INLINE__ std::string GetLTOIRMetadata(const std::string& filename) {
    std::string cache_dir = GetKernelCacheDirectory();
    if (cache_dir.empty()) {
      MATX_LOG_DEBUG("Cannot get metadata for {}: no cache directory", filename);
      return "";
    }
    
    try {
      std::filesystem::path meta_file = std::filesystem::path(cache_dir) / (filename + ".meta");
      if (!std::filesystem::exists(meta_file)) {
        MATX_LOG_DEBUG("No metadata file found for: {}", filename);
        return "";
      }
      
      std::ifstream file(meta_file);
      if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string metadata = buffer.str();
        MATX_LOG_DEBUG("Retrieved metadata for {}: {}", filename, metadata);
        return metadata;
      } else {
        MATX_LOG_ERROR("Failed to open metadata file: {}", meta_file.string());
      }
    } catch (const std::exception& e) {
      MATX_LOG_ERROR("Failed to retrieve metadata for {}: {}", filename, e.what());
    }
    return "";
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
    if (!data || size == 0) {
      MATX_LOG_ERROR("Cannot store empty data for: {}", filename);
      return false;
    }
    
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    try {
      // Create a new buffer and copy the data
      char* buffer = static_cast<char*>(malloc(size));
      if (!buffer) {
        MATX_LOG_ERROR("Failed to allocate {} bytes for: {}", size, filename);
        return false;
      }
      std::memcpy(buffer, data, size);
      
      // Store in the cache with size information
      ltoir_cache[filename] = LTOIRData{buffer, size};
      MATX_LOG_DEBUG("Stored {} bytes (copy) in memory cache for: {}", size, filename);
      
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
            MATX_LOG_DEBUG("Stored {} bytes (copy) to disk cache: {}", size, cache_file.string());
          } else {
            MATX_LOG_WARN("Failed to open file for writing: {}", cache_file.string());
          }
          // Note: We don't fail if disk write fails, as in-memory cache is still valid
        } catch (const std::exception& e) {
          MATX_LOG_WARN("Failed to write to disk cache for {}: {}", filename, e.what());
          // Ignore disk write failures - in-memory cache is still valid
        }
      }
      
      return true;
    } catch (const std::exception& e) {
      MATX_LOG_ERROR("Failed to store cache data for {}: {}", filename, e.what());
      // Handle any allocation or copy failures
      return false;
    }
  }

  /**
   * @brief Get the size of cached data
   * 
   * @param filename The key to check for
   * @return Size of the cached data in bytes, or 0 if not found
   */
  __MATX_INLINE__ size_t GetLTOIRCachedBytesLength(const std::string& filename) {
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      return it->second.length;
    }
    return 0;
  }

  /**
   * @brief Check if a key exists in the LTO IR cache
   * 
   * @param filename The key to check for
   * @return true if the key exists in the cache, false otherwise
   */
  __MATX_INLINE__ bool HasLTOIRCachedBytes(const std::string& filename) {
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    return ltoir_cache.find(filename) != ltoir_cache.end();
  }

  /**
   * @brief Remove an entry from the LTO IR cache
   * 
   * @param filename The key to remove
   * @return true if the key was found and removed, false if it didn't exist
   */
  __MATX_INLINE__ bool RemoveLTOIRCachedBytes(const std::string& filename) {
    [[maybe_unused]] std::lock_guard<std::recursive_mutex> lock(ltoir_mutex);
    
    auto it = ltoir_cache.find(filename);
    if (it != ltoir_cache.end()) {
      ltoir_cache.erase(it);
      return true;
    }
    return false;
  }


private:
  // Static cache for in-memory storage
  std::unordered_map<std::string, LTOIRData> ltoir_cache;
  std::unordered_map<CacheId, std::any> cache;
  std::unordered_map<CacheCommonParamsKey, std::unordered_map<cudaStream_t, StreamAllocation>, CacheCommonParamsKeyHash> stream_alloc_cache;
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
