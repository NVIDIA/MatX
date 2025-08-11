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

// This file is intended to contain simple defines that don't rely on any other
// MatX headers. It must be usable on both host and device compilers
#include <cuda/std/limits>

namespace matx {

#ifdef MATX_INDEX_32_BIT
    using index_t = int32_t;
    #define MATX_INDEX_T_FMT "d"
#else
    using index_t = long long int;
    #define MATX_INDEX_T_FMT "lld"
#endif

// By default, MatX opts out of additional handling of NaNs and infinite values
// in complex multiplication and division. These checks are defined in Annex G
// as an optional part of the C11 standard, but are not specified in the C++
// standard. The checks add additional cost beyond a "standard" implementation
// that computes, for example, complex multiplication via:
//   (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
// Users can opt-in to the additional checks by defining MATX_EN_COMPLEX_OP_NAN_CHECKS.
#ifndef MATX_EN_COMPLEX_OP_NAN_CHECKS
#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#endif

#ifdef __CUDACC__
    #define __MATX_HOST__ __host__
    #define __MATX_DEVICE__ __device__
#else
    #define __MATX_HOST__
    #define __MATX_DEVICE__
#endif

#ifdef __GNUC__
    #define __MATX_INLINE__ __attribute__((always_inline)) inline
#elif __CUDACC__
    #define __MATX_INLINE__ __forceinline__
#else
    #define __MATX_INLINE__ inline
#endif


#define MATX_STRINGIFY(x) #x
#define MATX_TOSTRING(x) MATX_STRINGIFY(x)

#if defined(__clang__ )
    #define MATX_IGNORE_WARNING_PUSH_GCC(WARN_MSG)
    #define MATX_IGNORE_WARNING_POP_GCC

    #define MATX_IGNORE_WARNING_PUSH_CLANG(WARN_MSG) \
        _Pragma("clang diagnostic push") \
        _Pragma(MATX_TOSTRING(clang diagnostic ignored WARN_MSG))

    #define MATX_IGNORE_WARNING_POP_CLANG \
        _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
    #define MATX_IGNORE_WARNING_PUSH_CLANG(WARN_MSG)
    #define MATX_IGNORE_WARNING_POP_CLANG

    #define MATX_IGNORE_WARNING_PUSH_GCC(WARN_MSG) \
        _Pragma("GCC diagnostic push") \
        _Pragma(MATX_TOSTRING(GCC diagnostic ignored WARN_MSG))

    #define MATX_IGNORE_WARNING_POP_GCC \
        _Pragma("GCC diagnostic pop")
#else
    #define MATX_IGNORE_WARNING_PUSH_GCC(WARN_MSG)
    #define MATX_IGNORE_WARNING_POP_GCC
    #define MATX_IGNORE_WARNING_PUSH_CLANG(WARN_MSG)
    #define MATX_IGNORE_WARNING_POP_CLANG
#endif

// std::ceil is not constexpr until C++23
#define MATX_ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))

enum {
  matxKeepDim     = cuda::std::numeric_limits<index_t>::max(),
  matxDropDim     = cuda::std::numeric_limits<index_t>::max() - 1,
  matxEnd         = cuda::std::numeric_limits<index_t>::max() - 2,
  matxKeepStride  = cuda::std::numeric_limits<index_t>::max() - 3,

  // If adding a new marker adjust this to the last element above
  matxIdxSentinel = matxKeepStride - 1,
};

// Do this on a per-architecture basis in the future
static constexpr int MAX_VEC_WIDTH_BYTES = 16;


#if defined(__CUDACC__)
  #define MATX_LOOP_UNROLL _Pragma("unroll")
  #define MATX_LOOP_DO_NOT_UNROLL _Pragma("unroll 1")
#else
#  define MATX_LOOP_UNROLL
  #define MATX_LOOP_DO_NOT_UNROLL
#endif

}
