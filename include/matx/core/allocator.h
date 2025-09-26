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


#include <cstdio>
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <utility>
#ifndef __CUDA_CC__
#include <driver_types.h>
#include <cuda_runtime_api.h>
#endif

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include <cuda/std/functional>
#include <cuda/std/__algorithm/max.h>

#pragma once

namespace matx {

/**
 * @brief Space where memory is stored (also called Kind in some contexts)
 * 
 */
enum matxMemorySpace_t {
  MATX_MANAGED_MEMORY,      ///< CUDA managed memory or CUDA Unified Memory (UM) from cudaMallocManaged
  MATX_HOST_MEMORY,         ///< CUDA host-pinned memory from cudaHostAlloc
  MATX_HOST_MALLOC_MEMORY,  ///< Host-alloced memory (pageable) from malloc
  MATX_DEVICE_MEMORY,       ///< CUDA device memory from cudaMalloc
  MATX_ASYNC_DEVICE_MEMORY, ///< CUDA asynchronous device memory corresponding to a stream from cudaMallocAsync
  MATX_INVALID_MEMORY       ///< Sentinel value
};

namespace detail {
struct matxMemoryStats_t {
  size_t currentBytesAllocated;
  size_t totalBytesAllocated;
  size_t maxBytesAllocated;
  matxMemoryStats_t()
      : currentBytesAllocated(0), totalBytesAllocated(0), maxBytesAllocated(0)
  {
  }
};

struct matxPointerAttr_t {
  size_t size;
  matxMemorySpace_t kind = MATX_INVALID_MEMORY;
  cudaStream_t stream;
};
}


inline detail::matxMemoryStats_t matxMemoryStats; ///< Statistics object
inline std::shared_mutex memory_mtx; ///< Mutex protecting updates from map

struct MemTracker {
  std::unordered_map<void *, detail::matxPointerAttr_t> allocationMap;

  auto size() {
    return allocationMap.size();
  }

  void update_stream(void *ptr, cudaStream_t stream) {
    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    auto iter = allocationMap.find(ptr);
    if (iter == allocationMap.end()) {
      MATX_THROW(matxInvalidParameter, "Couldn't find pointer in allocation cache");
    }

    iter->second.stream = stream;
  }

  template <typename StreamType>
  auto deallocate_internal(void *ptr, [[maybe_unused]] StreamType st) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    auto iter = allocationMap.find(ptr);

    if (iter == allocationMap.end()) {
  #ifdef MATX_DISABLE_MEM_TRACK_CHECK
      // This error can occur in situations where the user includes MatX in multiple translation units
      // and a deallocation occurs in a different one than it was allocated. Allow the user to ignore
      // these cases if they know the issue.
      MATX_THROW(matxInvalidParameter, "Couldn't find pointer in allocation cache");
  #else
      return;      
  #endif    
    }

    size_t bytes = iter->second.size;

    matxMemoryStats.currentBytesAllocated -= bytes;

    switch (iter->second.kind) {
    case MATX_MANAGED_MEMORY:
      [[fallthrough]];
    case MATX_DEVICE_MEMORY:
      cudaFree(ptr);
      break;
    case MATX_HOST_MEMORY:
      cudaFreeHost(ptr);
      break;
    case MATX_HOST_MALLOC_MEMORY:
      free(ptr);
      break;
    case MATX_ASYNC_DEVICE_MEMORY:
      if constexpr (std::is_same_v<no_stream_t, StreamType>) {
        cudaFreeAsync(ptr, iter->second.stream);
      }
      else {
        cudaFreeAsync(ptr, st.stream);
      }
      break;
    default:
      MATX_THROW(matxInvalidType, "Invalid memory type");
    }

    allocationMap.erase(ptr);    
  }

  struct no_stream_t{};
  struct valid_stream_t { cudaStream_t stream; };

  auto deallocate(void *ptr) {
    deallocate_internal(ptr, no_stream_t{});
  }

  auto deallocate(void *ptr, cudaStream_t stream) {
    deallocate_internal(ptr, valid_stream_t{stream});
  }    

  void allocate(void **ptr, size_t bytes,
                      matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                      cudaStream_t stream = 0) {
    [[maybe_unused]] cudaError_t err = cudaSuccess;
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    if (ptr == nullptr) {
      MATX_THROW(matxInvalidParameter, "nullptr on allocate");
    }

    *ptr = nullptr;

    // If requesting managed memory, check if the device supports concurrent managed access.
    // If not, fall back to pinned host memory. Jetsons are one system type where this is needed.
    if (space == MATX_MANAGED_MEMORY) {
      int device = 0;
      MATX_CUDA_CHECK(cudaGetDevice(&device));
      int concurrentManagedAccess = 0;
      MATX_CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, device));
      if (concurrentManagedAccess == 0) {
        space = MATX_HOST_MEMORY;
      }
    }
    
    switch (space) {
    case MATX_MANAGED_MEMORY:
      err = cudaMallocManaged(ptr, bytes);
      break;
    case MATX_HOST_MEMORY:
      err = cudaMallocHost(ptr, bytes);
      break;
    case MATX_HOST_MALLOC_MEMORY:
      *ptr = malloc(bytes);
      break;
    case MATX_DEVICE_MEMORY:
      err = cudaMalloc(ptr, bytes);
      break;
    case MATX_ASYNC_DEVICE_MEMORY:
      err = cudaMallocAsync(ptr, bytes, stream);
      break;
    case MATX_INVALID_MEMORY:
      MATX_THROW(matxInvalidType, "Invalid memory kind when allocating!");
    };

    MATX_ASSERT_STR_EXP(err, cudaSuccess, matxOutOfMemory, 
      "Failed to allocate memory. May be an asynchronous error from another CUDA call");

    if (*ptr == nullptr) {
      MATX_THROW(matxOutOfMemory, "Failed to allocate memory");
    }

    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    matxMemoryStats.currentBytesAllocated += bytes;
    matxMemoryStats.totalBytesAllocated += bytes;
    matxMemoryStats.maxBytesAllocated = cuda::std::max(
        matxMemoryStats.maxBytesAllocated, matxMemoryStats.currentBytesAllocated);
    allocationMap[*ptr] = {bytes, space, stream};
  }

  bool is_allocated(void *ptr) {
    if (ptr == nullptr) {
      return false;
    }

    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    auto iter = allocationMap.find(ptr);

    return iter != allocationMap.end();    
  }

  matxMemorySpace_t get_pointer_kind(void *ptr) {
    if (ptr == nullptr) {
      return MATX_INVALID_MEMORY;
    }

    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    auto iter = allocationMap.find(ptr);

    if (iter != allocationMap.end()) {
      return iter->second.kind;
    }

    return MATX_INVALID_MEMORY;
  }

  ~MemTracker() {
    while (allocationMap.size()) {
      deallocate(allocationMap.begin()->first);
    }
  }
};



__attribute__ ((visibility ("default")))
__MATX_INLINE__ MemTracker &GetAllocMap() {
  static MemTracker tracker;
  return tracker;
}

/**
 * @brief Determine if a pointer is printable by the host
 * 
 * Pointers are printable if they're either a managed or pinned memory pointer
 * 
 * @param mem Memory space
 * @return True is pointer can be printed from the host
 */
__MATX_INLINE__ bool HostPrintable(matxMemorySpace_t mem)
{
  return (mem == MATX_MANAGED_MEMORY || mem == MATX_HOST_MEMORY || mem == MATX_HOST_MALLOC_MEMORY);
}

/**
 * @brief Determine if a pointer is printable by the device
 * 
 * Pointers are printable if they're either a managed or device memory pointer
 * 
 * @param mem Memory space
 * @return True is pointer can be printed from the device
 */
__MATX_INLINE__ bool DevicePrintable(matxMemorySpace_t mem)
{
  return (mem == MATX_MANAGED_MEMORY || mem == MATX_DEVICE_MEMORY ||
          mem == MATX_ASYNC_DEVICE_MEMORY);
}

/**
 * @brief Get memory statistics
 * 
 * @param current Current memory usage
 * @param total Total memory usage
 * @param max Maximum memory usage
 */
__MATX_INLINE__ void matxGetMemoryStats(size_t *current, size_t *total, size_t *max)
{
  // std::shared_lock lck(memory_mtx);
  *current = matxMemoryStats.currentBytesAllocated;
  *total = matxMemoryStats.totalBytesAllocated;
  *max = matxMemoryStats.maxBytesAllocated;
}

/**
 * @brief Check if a pointer was allocated
 * 
 * @param ptr Pointer
 * @return True if allocator
 */
__MATX_INLINE__ bool IsAllocated(void *ptr) {
  return GetAllocMap().is_allocated(ptr);
}

/**
 * Get the kind of pointer based on an address
 *
 * Returns the memory kind of the pointer (device, host, managed, etc) based on
 *a pointer address. This function should not be used in the data path since it
 *takes a mutex and possibly loops through a std::map. Since Views can modify
 *the address of the data pointer, the base pointer may not be what is passed in
 * to this function, and therefore would not be in the map. However, finding the
 *next lowest address that is in the map is a good enough approximation since we
 *also offset in a positive direction from the base, and generally if you're in
 *a specific address range the type of pointer is obvious anyways.
 **/
__MATX_INLINE__ matxMemorySpace_t GetPointerKind(void *ptr)
{
  return GetAllocMap().get_pointer_kind(ptr);
}

/**
 * @brief Print memory statistics to stdout
 * 
 */
__MATX_INLINE__ void matxPrintMemoryStatistics()
{
  size_t current, total, max;

  matxGetMemoryStats(&current, &total, &max);

  printf("Memory Statistics(GB):  current: %.2f, total: %.2f, max: %.2f. Total "
         "allocations: %lu\n",
         static_cast<double>(current) / 1e9, static_cast<double>(total) / 1e9,
         static_cast<double>(max) / 1e9, GetAllocMap().size());
}

/**
 * @brief Allocate memory
 * 
 * Can be used for managed, pinned, malloced, device, and async device allocations
 * 
 * @param ptr Pointer to store allocated pointer
 * @param bytes Bytes to allocate
 * @param space Memory space
 * @param stream CUDA stream (for stream allocations)
 */
__MATX_INLINE__ void matxAlloc(void **ptr, size_t bytes,
                      matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                      cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  return GetAllocMap().allocate(ptr, bytes, space, stream);
}


__MATX_INLINE__ void matxFree(void *ptr)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  return GetAllocMap().deallocate(ptr);
}


__MATX_INLINE__ void matxFree(void *ptr, cudaStream_t stream)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  return GetAllocMap().deallocate(ptr, stream);
}

/**
  Update the stream a pointer in the cache is using. This should be used when the call wants to use 
  memory that was allocated in stream A inside of stream B. The caller must ensure that the pointer
  and stream being used are valid.
*/
__MATX_INLINE__ void update_stream(void *ptr, cudaStream_t stream)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  GetAllocMap().update_stream(ptr, stream);
}

/**
 * @brief Allocator following the PMR interface using the internal MatX allocator/deallocator
 * 
 */
template <typename T>
struct matx_allocator {
  friend void swap([[maybe_unused]] matx_allocator<T> &lhs, [[maybe_unused]] matx_allocator<T> &rhs) noexcept  { }   

  /**
   * @brief Allocate memory of at least ``size`` bytes
   * 
   * @param size Size of allocation in bytes
   * @return Pointer to allocated memory, or nullptr on error
   */
  __MATX_INLINE__ T* allocate(size_t size)
  {
    T *tmp;
    matxAlloc(reinterpret_cast<void**>(&tmp), size);
    return tmp;
  }

  /**
   * @brief Deallocate memory of at least ``size`` bytes
   * 
   * @param ptr Pointer to allocated data
   * @param size Size of previously-allocated memory in bytes
   */
  __MATX_INLINE__ void deallocate(void *ptr, [[maybe_unused]] size_t size)
  {
    matxFree(ptr);
  }  
};

__MATX_INLINE__ std::string SpaceString(matxMemorySpace_t space) {
  switch (space) {
    case MATX_MANAGED_MEMORY: return "CUDA managed memory";
    case MATX_HOST_MEMORY: return "CUDA host-pinned memory";
    case MATX_HOST_MALLOC_MEMORY: return "Host memory";
    case MATX_DEVICE_MEMORY: return "CUDA device memory";
    case MATX_ASYNC_DEVICE_MEMORY: return "CUDA asynchronous device memory";
    default: return "Unknown memory";
  }
}

} // end namespace matx
