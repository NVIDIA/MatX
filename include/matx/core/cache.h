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

#include "matx/core/error.h"

namespace matx {
namespace detail {

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

private:
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
