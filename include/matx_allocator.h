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

#include "matx_error.h"

#pragma once

namespace matx {

/**
 * @brief Space where memory is stored (also called Kind in some contexts)
 * 
 */
enum matxMemorySpace_t {
  MATX_MANAGED_MEMORY,
  MATX_HOST_MEMORY,
  MATX_HOST_MALLOC_MEMORY,
  MATX_DEVICE_MEMORY,
  MATX_ASYNC_DEVICE_MEMORY,
  MATX_INVALID_MEMORY
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
inline std::unordered_map<void *, detail::matxPointerAttr_t> allocationMap; ///< Map recording allocations

/**
 * @brief Determine if a pointer is printable by the host
 * 
 * Pointers are printable if they're either a managed or pinned memory pointer
 * 
 * @param mem Memory space
 * @return True is pointer can be printed from the host
 */
inline bool HostPrintable(matxMemorySpace_t mem)
{
  return (mem == MATX_MANAGED_MEMORY || mem == MATX_HOST_MEMORY);
}

/**
 * @brief Determine if a pointer is printable by the device
 * 
 * Pointers are printable if they're either a managed or device memory pointer
 * 
 * @param mem Memory space
 * @return True is pointer can be printed from the device
 */
inline bool DevicePrintable(matxMemorySpace_t mem)
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
inline void matxGetMemoryStats(size_t *current, size_t *total, size_t *max)
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
inline bool IsAllocated(void *ptr) {
  if (ptr == nullptr) {
    return false;
  }

  std::unique_lock lck(memory_mtx);
  auto iter = allocationMap.find(ptr);

  return iter != allocationMap.end();
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
inline matxMemorySpace_t GetPointerKind(void *ptr)
{
  if (ptr == nullptr) {
    return MATX_INVALID_MEMORY;
  }

  std::unique_lock lck(memory_mtx);
  auto iter = allocationMap.find(ptr);

  if (iter != allocationMap.end()) {
    return iter->second.kind;
  }


  return MATX_INVALID_MEMORY;
}

/**
 * @brief Print memory statistics to stdout
 * 
 */
inline void matxPrintMemoryStatistics()
{
  size_t current, total, max;

  matxGetMemoryStats(&current, &total, &max);

  printf("Memory Statistics(GB):  current: %.2f, total: %.2f, max: %.2f. Total "
         "allocations: %lu\n",
         static_cast<double>(current) / 1e9, static_cast<double>(total) / 1e9,
         static_cast<double>(max) / 1e9, allocationMap.size());
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
inline void matxAlloc(void **ptr, size_t bytes,
                      matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                      cudaStream_t stream = 0)
{
  [[maybe_unused]] cudaError_t err = cudaSuccess;

  switch (space) {
  case MATX_MANAGED_MEMORY:
    err = cudaMallocManaged(ptr, bytes);
    MATX_ASSERT(err == cudaSuccess, matxOutOfMemory);
    break;
  case MATX_HOST_MEMORY:
    err = cudaMallocHost(ptr, bytes);
    MATX_ASSERT(err == cudaSuccess, matxOutOfMemory);
    break;
  case MATX_HOST_MALLOC_MEMORY:
    *ptr = malloc(bytes);
    break;
  case MATX_DEVICE_MEMORY:
    err = cudaMalloc(ptr, bytes);
    MATX_ASSERT(err == cudaSuccess, matxOutOfMemory);
    break;
  case MATX_ASYNC_DEVICE_MEMORY:
    err = cudaMallocAsync(ptr, bytes, stream);
    MATX_ASSERT(err == cudaSuccess, matxOutOfMemory);
    break;
  case MATX_INVALID_MEMORY:
    MATX_THROW(matxInvalidType, "Invalid memory kind when allocating!");
    break;
  };
  
  MATX_ASSERT(ptr != nullptr, matxOutOfMemory);

  std::unique_lock lck(memory_mtx);
  matxMemoryStats.currentBytesAllocated += bytes;
  matxMemoryStats.totalBytesAllocated += bytes;
  matxMemoryStats.maxBytesAllocated = std::max(
      matxMemoryStats.maxBytesAllocated, matxMemoryStats.currentBytesAllocated);
  allocationMap[*ptr] = {bytes, space, stream};
}

/**
 * @brief Free previously-allocated pointer
 * 
 * @param ptr Pointer to free
 */
inline void matxFree(void *ptr)
{
  if (ptr == nullptr) {
    return;
  }

  std::unique_lock lck(memory_mtx);
  auto iter = allocationMap.find(ptr);

  if (iter == allocationMap.end()) {
    MATX_THROW(matxInvalidParameter, "Couldn't find pointer in allocation cache");
    return;
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
    cudaFreeAsync(ptr, iter->second.stream);
    break;
  default:
    MATX_THROW(matxInvalidType, "Invalid memory type");
  }

  allocationMap.erase(iter);
}

/**
 * @brief Allocator following the PMR interface using the internal MatX allocator/deallocator
 * 
 */
template <typename T>
struct matx_allocator {
  /**
   * @brief Allocate memory of at least ``size`` bytes
   * 
   * @param size Size of allocation in bytes
   * @return Pointer to allocated memory, or nullptr on error
   */
  inline T* allocate(size_t size)
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
  inline void deallocate(void *ptr, [[maybe_unused]] size_t size)
  {
    matxFree(ptr);
  }  
};

} // end namespace matx
