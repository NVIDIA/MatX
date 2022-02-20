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

#include "matx.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <regex>
#include <unordered_map>

namespace matx {

/**
 * @brief Utilities for unit tests
 * 
 */
class MatXUtils {
public:

  /**
   * @brief Compare two types with a specified delta
   * 
   * @tparam T1 First type
   * @tparam T2 Second type
   * @param a First value
   * @param b Second value
   * @param delta Delta between values
   * @return Assertion result
   */
  template <typename T1, typename T2>
  static __MATX_INLINE__ ::testing::AssertionResult MatXTypeCompare(const T1 &a, const T2 &b,
                                                    double delta = 0.01)
  {
    if constexpr (matx::is_complex_v<T1>) {
      if (fabs(static_cast<double>(a.real()) - static_cast<double>(b.real())) >
          delta) {
        printf("Real part failed in match: %f != %f\n",
               static_cast<double>(a.real()), static_cast<double>(b.real()));
        return ::testing::AssertionFailure();
      }
      if (fabs(static_cast<double>(a.imag()) - static_cast<double>(b.imag())) >
          delta) {
        printf("Imag part failed in match: %f != %f\n",
               static_cast<double>(a.imag()), static_cast<double>(b.imag()));
        return ::testing::AssertionFailure();
      }
    }
    else if constexpr (is_matx_half_v<T1> || is_half_v<T1>) {
      if (fabs(static_cast<float>(a) - static_cast<float>(b)) > delta) {
        std::cout << "Failed in match: " << static_cast<float>(a)
                  << " != " << static_cast<float>(b) << "\n";
        return ::testing::AssertionFailure();
      }
    }
    else if (fabs((double)a - (double)b) > delta) {
      std::cout << "Failed in match: " << a << " != " << b << "\n";
      return ::testing::AssertionFailure();
    }

    return ::testing::AssertionSuccess();
  }
};

template <typename T>
__MATX_INLINE__ void CheckTestTypeSupport() {
  auto cc = detail::GetComputeCapabilityMajor();
  if constexpr (is_bf16_type_v<T>) {
    if (cc < AMPERE_CC) {
      GTEST_SKIP();
    }
  }
  else if constexpr (is_fp16_type_v<T>) {
    if (cc < PASCAL_CC) {
      GTEST_SKIP();
    }
  }
}

template <typename T>
__MATX_INLINE__ void CheckTestTensorCoreTypeSupport() {
  auto cc = detail::GetComputeCapabilityMajor();
  if constexpr (is_bf16_type_v<T>) {
    if (cc < AMPERE_CC) {
      GTEST_SKIP();
    }
  }
  else if constexpr (is_fp16_type_v<T>) {
    if (cc < VOLTA_CC) {
      GTEST_SKIP();
    }
  }
}

} // end namespace matx
