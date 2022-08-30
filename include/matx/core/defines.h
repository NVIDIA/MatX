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

// This file is intended to contain simple defines that don't rely on any other headers. It must be
// useable on both host and device compilers

namespace matx {

#ifdef INDEX_64_BIT
    using index_t = long long int;
#endif

#ifdef INDEX_32_BIT
    using index_t = int32_t;
#endif

#if ((defined(INDEX_64_BIT) && defined(INDEX_32_BIT)) ||                       \
     (!defined(INDEX_64_BIT) && !defined(INDEX_32_BIT)))
static_assert(false, "Must choose either 64-bit or 32-bit index mode");
#endif

#ifdef __CUDACC__
    #define __MATX_HOST__ __host__
    #define __MATX_DEVICE__ __device__
#else
    #define __MATX_HOST__  __host__
    #define __MATX_DEVICE__ __device__
#endif

#ifdef __GNUC__ 
    #define __MATX_INLINE__ __attribute__((always_inline)) inline
#elif __CUDACC__
    #define __MATX_INLINE__ __forceinline__ 
#else
    #define __MATX_INLINE__ inline
#endif

}