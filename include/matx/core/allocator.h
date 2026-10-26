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
#include <cuda.h>

#include "matx/core/device_attributes.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/log.h"
#include "matx/core/resource_guard.h"
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
  MATX_ASYNC_DEVICE_MEMORY, ///< Stream-ordered device memory from cudaMallocAsync, or cudaMalloc when memory pools are unsupported
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

// What the stream-device lookup needs to know about the process. None of it
// can change while the process runs, so it is resolved once.
//
// The lookup reads the stream's context rather than calling cuStreamGetDevice
// (12.8+ drivers). Newer drivers allow cuStreamGetDevice during CUDA graph
// capture, but a range of drivers, including the 595 branch, reject it on a
// capturing stream and invalidate the capture. The context queries work during
// capture on every supported driver.
struct StreamDeviceLookup {
  bool single_device = false;
  // cuStreamGetCtx_v2, on 12.5+ drivers. Unlike cuStreamGetCtx, it accepts
  // green-context streams. The last parameter is a CUgreenCtx *, which
  // toolkits before 12.4 do not declare.
  CUresult (CUDAAPI *get_ctx_v2)(CUstream, CUcontext *, void **) = nullptr;
};

inline StreamDeviceLookup MakeStreamDeviceLookup() {
  StreamDeviceLookup lookup;
  lookup.single_device = GetDeviceCount() == 1;
  // cuGetProcAddress returns the newest version of a symbol that the driver
  // has, so an older driver would return the two-argument cuStreamGetCtx.
  int version = 0;
  if (cuDriverGetVersion(&version) == CUDA_SUCCESS && version >= 12050) {
    void *pfn = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (cuGetProcAddress("cuStreamGetCtx", &pfn, 12050, CU_GET_PROC_ADDRESS_DEFAULT,
                         &status) == CUDA_SUCCESS &&
        status == CU_GET_PROC_ADDRESS_SUCCESS) {
      lookup.get_ctx_v2 = reinterpret_cast<decltype(lookup.get_ctx_v2)>(pfn);
    }
  }
  MATX_LOG_DEBUG("Stream device lookup: driver {}, cuStreamGetCtx_v2 {}",
                 version, lookup.get_ctx_v2 != nullptr);
  return lookup;
}

inline const StreamDeviceLookup &GetStreamDeviceLookup() {
  static const StreamDeviceLookup lookup = MakeStreamDeviceLookup();
  return lookup;
}

// Device that owns a stream. cudaMallocAsync allocates from this device's
// memory pool, which can differ from the current device. Tests pass a lookup
// to take each branch.
inline int GetStreamDevice(cudaStream_t stream,
                           const StreamDeviceLookup &lookup = GetStreamDeviceLookup()) {
  if (lookup.single_device) {
    return 0;
  }
  // The default-stream handles name the current device's default stream. The
  // driver resolves them through the current context, which a thread may not
  // have yet.
  if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
    int device = 0;
    MATX_CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }
  // Read the stream's context. cuStreamGetCtx rejects green-context streams;
  // cuStreamGetCtx_v2 returns the primary context of their device.
  const auto cu_stream = reinterpret_cast<CUstream>(stream);
  CUcontext ctx = nullptr;
  if (lookup.get_ctx_v2 != nullptr) {
    void *green_ctx = nullptr;
    MATX_CUDA_DRIVER_CHECK(lookup.get_ctx_v2(cu_stream, &ctx, &green_ctx));
  } else {
    MATX_CUDA_DRIVER_CHECK(cuStreamGetCtx(cu_stream, &ctx));
  }
  // cuCtxGetDevice reads the current context, so make ctx current briefly.
  MATX_CUDA_DRIVER_CHECK(cuCtxPushCurrent(ctx));
  CUdevice device = 0;
  const CUresult err = cuCtxGetDevice(&device);
  MATX_CUDA_DRIVER_CHECK(cuCtxPopCurrent(nullptr));
  MATX_CUDA_DRIVER_CHECK(err);
  return static_cast<int>(device);
}

// Signature of TryGetDeviceAttributes, so tests can inject query failures.
using TryGetDeviceAttributesFn = cudaError_t (*)(int, const DeviceAttributes *&);

// Whether every visible device supports memory pools. If so, cudaMallocAsync
// works on any stream, so allocations skip finding the stream's device. A
// device whose attributes cannot be read counts as lacking pools, so
// allocations check only their stream's device rather than all failing.
inline bool QueryAllDevicesSupportMemoryPools(
    TryGetDeviceAttributesFn try_get_attrs = &TryGetDeviceAttributes) {
  const int count = GetDeviceCount();
  for (int device = 0; device < count; ++device) {
    const DeviceAttributes *attrs = nullptr;
    const cudaError_t err = try_get_attrs(device, attrs);
    if (err != cudaSuccess) {
      // Clear the error so a later CUDA error check does not report it.
      cudaGetLastError();
      MATX_LOG_WARN("Device {} attribute query failed ({}). Checking memory pool support per allocation",
                    device, cudaGetErrorString(err));
      return false;
    }
    if (!attrs->memory_pools_supported) {
      return false;
    }
  }
  return true;
}

// Device attributes cannot change while the process runs, so this is checked
// once.
inline bool AllDevicesSupportMemoryPools() {
  static const bool all_supported = QueryAllDevicesSupportMemoryPools();
  return all_supported;
}
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

  // deallocate_internal assumes that the caller has already acquired the memory_mtx mutex.
  template <typename StreamType>
  auto deallocate_internal(void *ptr, [[maybe_unused]] StreamType st) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

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

    MATX_LOG_DEBUG("Deallocating memory: ptr={}, {} bytes, space={}, remaining={} bytes",
                   ptr, bytes, static_cast<int>(iter->second.kind), matxMemoryStats.currentBytesAllocated - bytes);

    matxMemoryStats.currentBytesAllocated -= bytes;

    // Check if the CUDA context is still valid before attempting to free.
    // During static destruction at program exit, the CUDA context may already
    // be destroyed, making cudaFree/cudaFreeAsync calls fail with
    // CUDA_ERROR_CONTEXT_IS_DESTROYED.
    auto is_cuda_free = [&]() {
      int dev;
      return cudaGetDevice(&dev) == cudaSuccess;
    };

    switch (iter->second.kind) {
    case MATX_MANAGED_MEMORY:
      [[fallthrough]];
    case MATX_DEVICE_MEMORY:
      if (is_cuda_free()) MATX_CUDA_CHECK_NOEXCEPT(cudaFree(ptr));
      break;
    case MATX_HOST_MEMORY:
      if (is_cuda_free()) MATX_CUDA_CHECK_NOEXCEPT(cudaFreeHost(ptr));
      break;
    case MATX_HOST_MALLOC_MEMORY:
      free(ptr);
      break;
    case MATX_ASYNC_DEVICE_MEMORY:
      if (is_cuda_free()) {
        if constexpr (cuda::std::is_same_v<no_stream_t, StreamType>) {
          MATX_CUDA_CHECK_NOEXCEPT(cudaFreeAsync(ptr, iter->second.stream));
        }
        else {
          MATX_CUDA_CHECK_NOEXCEPT(cudaFreeAsync(ptr, st.stream));
        }
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
    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    deallocate_internal(ptr, no_stream_t{});
  }

  auto deallocate(void *ptr, cudaStream_t stream) {
    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    deallocate_internal(ptr, valid_stream_t{stream});
  }

  void allocate(void **ptr, size_t bytes,
                      matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                      cudaStream_t stream = 0) {
    // Only async allocations need pool support, so other memory spaces never
    // query devices. Host allocations then work without a visible GPU.
    allocate_internal(ptr, bytes, space, stream,
        [](int device) { return detail::GetDeviceAttributes(device).memory_pools_supported; },
        space == MATX_ASYNC_DEVICE_MEMORY && detail::AllDevicesSupportMemoryPools());
  }

  // Inject the capability query so tests can exercise the fallback on devices
  // with memory pools without overriding global state or changing matxAlloc.
  // all_devices_support_pools skips the stream's device lookup and the query.
  template <typename MemoryPoolSupport>
  void allocate_internal(void **ptr, size_t bytes, matxMemorySpace_t space,
                         cudaStream_t stream, MemoryPoolSupport supports_memory_pools,
                         bool all_devices_support_pools = false) {
    [[maybe_unused]] cudaError_t err = cudaSuccess;

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    if (ptr == nullptr) {
      MATX_THROW(matxInvalidParameter, "nullptr on allocate");
    }

    *ptr = nullptr;

    // If requesting managed memory, check if the device supports concurrent managed access.
    // If not, fall back to pinned host memory. Jetsons are one system type where this is needed.
    if (space == MATX_MANAGED_MEMORY) {
      if (!detail::GetCurrentDeviceAttributes().concurrent_managed_access) {
        space = MATX_HOST_MEMORY;
      }
    }

    // Devices without memory pools cannot use cudaMallocAsync. Record the
    // fallback kind so deallocation also uses cudaFree instead of cudaFreeAsync.
    // When every device has pools, the common case, cudaMallocAsync finds the
    // stream's device itself.
    int fallback_device = -1;
    if (space == MATX_ASYNC_DEVICE_MEMORY && !all_devices_support_pools) {
      const int stream_device = detail::GetStreamDevice(stream);
      if (!supports_memory_pools(stream_device)) {
        space = MATX_DEVICE_MEMORY;
        fallback_device = stream_device;
      }
    }

    MATX_LOG_DEBUG("Allocating memory: {} bytes, space={}, stream={}", bytes, static_cast<int>(space), reinterpret_cast<void*>(stream));

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
      if (fallback_device >= 0) {
        // cudaMalloc allocates on the current device, but the fallback belongs
        // on the stream's device, where cudaMallocAsync would have put it.
        detail::CudaDeviceGuard guard{fallback_device};
        err = cudaMalloc(ptr, bytes);
      }
      else {
        err = cudaMalloc(ptr, bytes);
      }
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

    MATX_LOG_DEBUG("Allocated memory: ptr={}, {} bytes, total_current={} bytes", *ptr, bytes, matxMemoryStats.currentBytesAllocated + bytes);

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

  void free_all() {
    [[maybe_unused]] std::unique_lock lck(memory_mtx);
    while (! allocationMap.empty()) {
      auto it = allocationMap.begin();
      const auto ptr = it->first;
      deallocate_internal(ptr, no_stream_t{});
      if (allocationMap.find(ptr) != allocationMap.end()) {
        // deallocate_internal may have erased the pointer from the map
        // If not, erase it here to avoid an infinite loop.
        allocationMap.erase(ptr);
      }
    }
  }

  ~MemTracker() {
    free_all();
  }
};



MATX_VISIBILITY_DEFAULT
__MATX_INLINE__ MemTracker &GetAllocMap() {
  static MemTracker tracker;
  return tracker;
}

// Helper function to free all MatX allocations. This function frees all allocations
// made with matxAlloc. These allocations may have been made directly by the user or they
// may have been made by MatX internally for workspaces. This function does not free the
// caches (i.e., allocations made for FFT plans, cuBLAS handles, and other state required
// for MatX transforms). To free those caches, use matx::ClearCaches(). It is not safe to
// call matxFree() on user-managed pointers after calling this function. This function should
// be called after the user application has called matxFree() on any pointers for which it
// will call matxFree().
MATX_VISIBILITY_DEFAULT
__MATX_INLINE__ void FreeAllocations() {
  GetAllocMap().free_all();
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
 * a pointer address. This function should not be used in the data path since it
 * takes a mutex.
 *
 * The lookup is an exact match against the address the allocator recorded, so the
 * base pointer of the allocation must be passed in. Views can offset their data
 * pointer into the allocation, and such a pointer is never found in the map, so
 * MATX_INVALID_MEMORY is returned for it.
 *
 * Note: an earlier version of this comment described an approximate lookup that
 * fell back to the next lowest recorded address. No such fallback exists here.
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
         "allocations: %zu\n",
         static_cast<double>(current) / 1e9, static_cast<double>(total) / 1e9,
         static_cast<double>(max) / 1e9, GetAllocMap().size());
}

/**
 * @brief Allocate memory
 *
 * Can be used for managed, pinned, malloced, device, and async device allocations
 * Async device allocations fall back to cudaMalloc/cudaFree, on the stream's
 * device, when that device does not support memory pools. GetPointerKind reports
 * MATX_DEVICE_MEMORY for these allocations, and allocation and deallocation may
 * synchronize.
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
