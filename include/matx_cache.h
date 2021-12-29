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

#include "matx_error.h"
#include <functional>
#include <optional>
#include <unordered_map>

namespace matx {
namespace detail {

/**
 * Generic caching object for caching parameters. This class is used for
 * creating handles/plans on-the-fly and caching them to remove the need for
 * plans on certain interfaces. For example, InParams can be all parameters
 * needed to define an FFT, and if that plan already exists, a user doesn't need
 * to create another plan.
 */
template <typename InParams, typename KeyHash, typename KeyEq>
class matxCache_t {
public:
  matxCache_t() {}

  /**
   * Look up parameters in the cache
   *
   * @param in
   *   Input parameters
   * @returns
   *   nullopt if no match exists, or the object request if exists
   */
  std::optional<void *> Lookup(InParams &in)
  {
    auto el = cache.find(in);
    if (el == cache.end()) {
      return std::nullopt;
    }

    return el->second;
  }

  /**
   * Insert an object into the cache
   *
   * @param params
   *   Input parameters (key)
   * @param obj
   *   Object to store (value)
   *
   */
  void Insert(InParams &params, void *obj) { cache.insert({params, obj}); }

  /**
   * Deletes the entire contents of the cache
   *
   */
  void Clear() { cache.clear(); }

private:
  std::unordered_map<InParams, void *, KeyHash, KeyEq> cache;
};

/**
 * Converts elements in a POD container to a hash value
 */
template <typename T, int len>
inline size_t PodArrayToHash(std::array<T, len> c)
{
  size_t hash = 0;
  for (auto &el : c) {
    hash += std::hash<T>()(el);
  }

  return hash;
}

}  // namespace detail
}; // namespace matx
