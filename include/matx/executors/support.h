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

#include "matx/core/type_utils.h"

// Utility functions to determine what support is available per-executor

namespace matx {
  namespace detail {

// FFT
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
  #define MATX_EN_CPU_FFT 1
#else
  #define MATX_EN_CPU_FFT 0
#endif

// MatMul
#if defined(MATX_EN_NVPL) || defined(MATX_EN_OPENBLAS) || defined(MATX_EN_BLIS)
  #define MATX_EN_CPU_MATMUL 1
#else
  #define MATX_EN_CPU_MATMUL 0
#endif  

// Solver
#if defined(MATX_EN_NVPL) || defined(MATX_EN_OPENBLAS_LAPACK)
  #define MATX_EN_CPU_SOLVER 1
#else
  #define MATX_EN_CPU_SOLVER 0
#endif

template <typename Exec, typename T>
constexpr bool CheckFFTSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    if constexpr (is_complex_half_v<T>) {
      return false;
    } else {
      return MATX_EN_CPU_FFT;
    }
  }
  else {
    return true;
  }
}

template <typename Exec>
constexpr bool CheckDirect1DConvSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    return false;
  }
  else {
    return true;
  }
}

template <typename Exec, typename T>
constexpr bool CheckFFT1DConvSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    return CheckFFTSupport<Exec, T>();
  }
  else {
    return true;
  }
}

template <typename Exec>
constexpr bool Check2DConvSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    return false;
  }
  else {
    return true;
  }
}

template <typename Exec, typename T>
constexpr bool CheckMatMulSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    if constexpr (std::is_same_v<T, float> ||
                  std::is_same_v<T, double> ||
                  std::is_same_v<T, cuda::std::complex<float>> ||
                  std::is_same_v<T, cuda::std::complex<double>>) {
      return MATX_EN_CPU_MATMUL;
    } else {
      return false;
    }
  }
  else {
    return true;
  }
}

template <typename Exec>
constexpr bool CheckSolverSupport() {
  if constexpr (is_host_executor_v<Exec>) {
    return MATX_EN_CPU_SOLVER;
  } else {
    return true;
  }
}

}; // detail
}; // matx