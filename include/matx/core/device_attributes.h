////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "matx/core/defines.h"
#include "matx/core/error.h"
#include "matx/core/log.h"

namespace matx {
namespace detail {

/**
 * @brief Device attributes that are fixed for the lifetime of the process.
 *
 * Only add attributes that cannot change while the process runs. Clock rates,
 * compute mode, and memory usage can change and must be queried directly. So
 * can the SM count: under MPS per-context partitioning,
 * cudaDevAttrMultiProcessorCount reports the share of the context current to
 * the calling thread.
 */
struct DeviceAttributes {
  int cc_major;                    ///< cudaDevAttrComputeCapabilityMajor
  int cc_minor;                    ///< cudaDevAttrComputeCapabilityMinor
  int max_smem_per_block;          ///< cudaDevAttrMaxSharedMemoryPerBlock
  int max_smem_per_block_optin;    ///< cudaDevAttrMaxSharedMemoryPerBlockOptin
  int max_smem_per_sm;             ///< cudaDevAttrMaxSharedMemoryPerMultiprocessor
  int max_regs_per_sm;             ///< cudaDevAttrMaxRegistersPerMultiprocessor
  int l2_cache_bytes;              ///< cudaDevAttrL2CacheSize
  int memory_bus_width_bits;       ///< cudaDevAttrGlobalMemoryBusWidth
  int fp32_to_fp64_perf_ratio;     ///< cudaDevAttrSingleToDoublePrecisionPerfRatio
  bool concurrent_managed_access;  ///< cudaDevAttrConcurrentManagedAccess
  bool memory_pools_supported;     ///< cudaDevAttrMemoryPoolsSupported

  /** Compute capability as __CUDA_ARCH__ digits (e.g., 890 for SM 8.9) */
  int ComputeCapability() const { return cc_major * 100 + cc_minor * 10; }
};

/** Signature of cudaDeviceGetAttribute, so tests can inject query failures */
using DeviceAttrQueryFn = cudaError_t (*)(int *, cudaDeviceAttr, int);

/**
 * @brief Query every DeviceAttributes field for a device directly from CUDA.
 *
 * A failed query of a required attribute fails the call. Optional attributes
 * only steer heuristics or fallbacks, so a failed query sets them to 0, which
 * selects the conservative path: no memory pools, no concurrent managed access,
 * and no L2, bandwidth, or FP64-rate tuning.
 *
 * @param device Device ordinal
 * @param attrs Filled on success; partially written on failure
 * @param query_attr Attribute query function, cudaDeviceGetAttribute by default
 * @return cudaSuccess, or the error from the first failed required query
 */
inline cudaError_t QueryDeviceAttributes(int device, DeviceAttributes &attrs,
                                         DeviceAttrQueryFn query_attr = &cudaDeviceGetAttribute) {
  int concurrent_managed_access = 0;
  int memory_pools_supported = 0;
  const struct {
    cudaDeviceAttr attr;
    int *value;
    bool required;
  } queries[] = {
    {cudaDevAttrComputeCapabilityMajor, &attrs.cc_major, true},
    {cudaDevAttrComputeCapabilityMinor, &attrs.cc_minor, true},
    {cudaDevAttrMaxSharedMemoryPerBlock, &attrs.max_smem_per_block, true},
    {cudaDevAttrMaxSharedMemoryPerBlockOptin, &attrs.max_smem_per_block_optin, true},
    {cudaDevAttrMaxSharedMemoryPerMultiprocessor, &attrs.max_smem_per_sm, true},
    {cudaDevAttrMaxRegistersPerMultiprocessor, &attrs.max_regs_per_sm, true},
    {cudaDevAttrL2CacheSize, &attrs.l2_cache_bytes, false},
    {cudaDevAttrGlobalMemoryBusWidth, &attrs.memory_bus_width_bits, false},
    {cudaDevAttrSingleToDoublePrecisionPerfRatio, &attrs.fp32_to_fp64_perf_ratio, false},
    {cudaDevAttrConcurrentManagedAccess, &concurrent_managed_access, false},
    {cudaDevAttrMemoryPoolsSupported, &memory_pools_supported, false},
  };

  for (const auto &query : queries) {
    const cudaError_t err = query_attr(query.value, query.attr, device);
    if (err == cudaSuccess) {
      continue;
    }
    if (query.required) {
      return err;
    }
    *query.value = 0;
    // Clear the tolerated error so a later CUDA error check does not report it.
    cudaGetLastError();
    MATX_LOG_WARN("Device attribute {} query failed on device {} ({}). Using 0",
                  static_cast<int>(query.attr), device, cudaGetErrorString(err));
  }

  attrs.concurrent_managed_access = concurrent_managed_access != 0;
  attrs.memory_pools_supported = memory_pools_supported != 0;
  return cudaSuccess;
}

/**
 * @brief Process-wide cache of DeviceAttributes, one entry per device ordinal.
 *
 * Each device is queried on its first lookup. Later lookups are lock-free.
 * A failed query of a required attribute is not cached, so a later lookup
 * retries. Failed optional attributes are stored as 0.
 */
class DeviceAttributeCache {
public:
  /**
   * @brief Look up the attributes for a device, querying CUDA on first use.
   *
   * @param device Device ordinal
   * @param attrs Set to the cached entry on success; unchanged on failure
   * @return cudaSuccess, cudaErrorInvalidDevice for an out-of-range ordinal,
   * or the CUDA error from the failed query
   */
  cudaError_t Get(int device, const DeviceAttributes *&attrs) {
    int count = 0;
    cudaError_t err = EnsureSlots(count);
    if (err != cudaSuccess) {
      return err;
    }
    if (device < 0 || device >= count) {
      return cudaErrorInvalidDevice;
    }

    Slot &slot = slots_[static_cast<size_t>(device)];
    if (!slot.ready.load(std::memory_order_acquire)) {
      std::lock_guard lock(slot.mtx);
      if (!slot.ready.load(std::memory_order_relaxed)) {
        DeviceAttributes queried{};
        err = QueryDeviceAttributes(device, queried);
        if (err != cudaSuccess) {
          return err;
        }
        slot.attrs = queried;
        slot.ready.store(true, std::memory_order_release);
        MATX_LOG_DEBUG("Cached attributes for device {}: compute capability {}.{}",
                       device, queried.cc_major, queried.cc_minor);
      }
    }

    attrs = &slot.attrs;
    return cudaSuccess;
  }

  /**
   * @brief Number of visible devices, queried from CUDA on first use.
   *
   * @param count Set to the device count on success
   * @return cudaSuccess or the error from cudaGetDeviceCount
   */
  cudaError_t DeviceCount(int &count) {
    return EnsureSlots(count);
  }

private:
  struct Slot {
    std::atomic<bool> ready{false};
    std::mutex mtx;
    DeviceAttributes attrs{};
  };

  // The device count is fixed once CUDA initializes, so the slots are
  // allocated on the first successful count and never resized.
  cudaError_t EnsureSlots(int &count) {
    count = device_count_.load(std::memory_order_acquire);
    if (count >= 0) {
      return cudaSuccess;
    }

    std::lock_guard lock(init_mtx_);
    count = device_count_.load(std::memory_order_relaxed);
    if (count >= 0) {
      return cudaSuccess;
    }

    int queried = 0;
    const cudaError_t err = cudaGetDeviceCount(&queried);
    if (err != cudaSuccess) {
      return err;
    }
    slots_ = std::make_unique<Slot[]>(static_cast<size_t>(queried));
    device_count_.store(queried, std::memory_order_release);
    count = queried;
    return cudaSuccess;
  }

  std::mutex init_mtx_;
  std::atomic<int> device_count_{-1};
  std::unique_ptr<Slot[]> slots_;
};

MATX_VISIBILITY_DEFAULT
inline DeviceAttributeCache &GetDeviceAttributeCache() {
  static DeviceAttributeCache cache;
  return cache;
}

/**
 * @brief Non-throwing lookup of the cached attributes for a device
 *
 * @param device Device ordinal
 * @param attrs Set to the cached entry on success; unchanged on failure
 * @return cudaSuccess or the CUDA error from the failed lookup
 */
inline cudaError_t TryGetDeviceAttributes(int device, const DeviceAttributes *&attrs) {
  return GetDeviceAttributeCache().Get(device, attrs);
}

/**
 * @brief Non-throwing lookup of the cached attributes for the calling thread's
 * current device
 *
 * With one visible device, the current device can only be device 0, so the
 * lookup skips cudaGetDevice.
 *
 * @param attrs Set to the cached entry on success; unchanged on failure
 * @return cudaSuccess, or the error from the device count, cudaGetDevice, or
 * the lookup
 */
inline cudaError_t TryGetCurrentDeviceAttributes(const DeviceAttributes *&attrs) {
  auto &cache = GetDeviceAttributeCache();
  int count = 0;
  cudaError_t err = cache.DeviceCount(count);
  if (err != cudaSuccess) {
    return err;
  }
  int device = 0;
  if (count != 1) {
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cache.Get(device, attrs);
}

/**
 * @brief Number of visible CUDA devices. Throws matxCudaError on failure.
 *
 * The count cannot change while the process runs, so it is read once.
 */
__MATX_INLINE__ int GetDeviceCount() {
  int count = 0;
  MATX_CUDA_CHECK(GetDeviceAttributeCache().DeviceCount(count));
  return count;
}

/**
 * @brief Cached attributes for a device. Throws matxCudaError on failure.
 *
 * @param device Device ordinal
 * @return Reference that stays valid until static destruction at process exit
 */
__MATX_INLINE__ const DeviceAttributes &GetDeviceAttributes(int device) {
  const DeviceAttributes *attrs = nullptr;
  MATX_CUDA_CHECK(TryGetDeviceAttributes(device, attrs));
  return *attrs;
}

/**
 * @brief Cached attributes for the calling thread's current device. Throws
 * matxCudaError on failure.
 *
 * @return Reference that stays valid until static destruction at process exit
 */
__MATX_INLINE__ const DeviceAttributes &GetCurrentDeviceAttributes() {
  const DeviceAttributes *attrs = nullptr;
  MATX_CUDA_CHECK(TryGetCurrentDeviceAttributes(attrs));
  return *attrs;
}

} // namespace detail
} // namespace matx
