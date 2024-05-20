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
#include <unordered_map>
#include <cuda/atomic>

#include "matx/core/error.h"

namespace matx {
namespace detail {

using CacheId = uint64_t;

inline cuda::std::atomic<CacheId> CacheIdCounter{0};

template<typename CacheType>
CacheId GetCacheIdFromType()
{
  static CacheId id = CacheIdCounter.fetch_add(1);

  return id;
}

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
    auto el = cache.find(id);
    MATX_ASSERT_STR(el != cache.end(), matxInvalidType, "Cache type not found");

    std::any_cast<CacheType>(el->second).clear();
  }

  template <typename CacheType, typename InParams, typename MakeFun, typename ExecFun>
  void LookupAndExec(const CacheId &id, const InParams &params, const MakeFun &mfun, const ExecFun &efun) {
    // Create named cache if it doesn't exist
    auto el = cache.find(id);
    if (el == cache.end()) {
      cache[id] = CacheType{};
    }

    auto &cval = cache[id];
    auto &rmap = std::any_cast<CacheType&>(cval);
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

private:
  std::unordered_map<CacheId, std::any> cache;
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
