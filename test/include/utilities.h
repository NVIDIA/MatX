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

#ifdef _MSC_VER
inline int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite) {
    size_t requiredSize;
    getenv_s(&requiredSize, nullptr, 0, name);
    if (requiredSize != 0) {
      return 0; // Variable already exists and overwrite is not allowed
    }
  }
  return _putenv_s(name, value);
}
inline int unsetenv(const char* name) {
  // _putenv with "NAME=" (no value after '=') removes the variable
  std::string var = std::string(name) + "=";
  return _putenv(var.c_str());
}
#endif

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

/**
 * @brief Number of visible CUDA devices, or 0 when CUDA reports an error
 *
 * Clears the error so later CUDA checks in the test do not report it.
 */
inline int CudaDeviceCount() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return device_count;
}

/**
 * @brief Create a driver API context on a device and make it current, as an
 * application that manages its own contexts would. Destroy it with cuCtxDestroy.
 */
inline CUcontext CreateDriverContext(int device) {
  CUcontext ctx = nullptr;
#if CUDA_VERSION >= 13000
  MATX_CUDA_DRIVER_CHECK(cuCtxCreate(&ctx, nullptr, 0, device));
#else
  MATX_CUDA_DRIVER_CHECK(cuCtxCreate(&ctx, 0, device));
#endif
  return ctx;
}

/** @brief The calling thread's current context, or nullptr if none is current */
inline CUcontext CurrentDriverContext() {
  CUcontext ctx = nullptr;
  MATX_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&ctx));
  return ctx;
}

} // end namespace matx
