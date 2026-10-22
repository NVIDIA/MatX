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

#include "matx.h"
#include <format>
#include <mutex>
#include <string>
#include <unordered_map>

#ifdef MATX_EN_JIT
namespace matx {

/**
 * Checks one JIT launch against every launch made earlier in the test process
 *
 * Compiled kernels are cached by the class names of their operators, or by a fixed-size cache key when every
 * operator in the expression provides one, together with the launch configuration. Two launches with the same
 * launch configuration that agree on a class name must therefore generate the same definition for it, and two
 * launches that agree on a cache key must have the same type string. Otherwise one of them would run the other's
 * kernel. A mismatch throws, which fails the test that made the launch.
 *
 * @param classes Class definitions of the launched expression, keyed by class name
 * @param jit_cache_key Fixed-size cache key of the expression, if it has a valid one
 * @param type_str Type string of the expression
 * @param launch_config Kernel variant, elements per thread, block size, and groups per block of the launch
 */
inline void CheckJITLaunchClassNames(const std::unordered_map<std::string, std::string> &classes,
                                     const detail::JITCacheKey &jit_cache_key, const std::string &type_str,
                                     const std::string &launch_config)
{
  static std::unordered_map<std::string, std::string> definitions;
  static std::unordered_map<detail::JITCacheKey, std::string, detail::JITCacheKeyHash> type_strings;
  static std::mutex mutex;

  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &[name, definition] : classes) {
    // Some operators generate different code for different launch configurations, so compare within one
    detail::register_jit_class(definitions, name + " (launch " + launch_config + ")", definition);
  }
  if (jit_cache_key.valid) {
    const auto [it, inserted] = type_strings.try_emplace(jit_cache_key, type_str);
    if (!inserted && it->second != type_str) {
      MATX_THROW(matxInvalidParameter, std::format("JIT cache key {} is shared by {} and {}",
                                                   detail::JITCacheKeyToHex(jit_cache_key), it->second, type_str));
    }
  }
}

/**
 * Makes every later JIT launch in the process run CheckJITLaunchClassNames
 */
inline void InstallJITClassNameCheck()
{
  detail::jit_launch_observer().store(CheckJITLaunchClassNames);
}

} // namespace matx
#endif
