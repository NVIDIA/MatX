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

#include "matx_half.h"

namespace matx {
namespace detail {

// This file defines derivatives for scalar operations
namespace { 
  using cuda::std::ceil;
  using cuda::std::floor;
  using cuda::std::round;
  using cuda::std::sqrt;
  using cuda::std::exp;
  using cuda::std::log10;
  using cuda::std::log2;
  using cuda::std::log;
  using cuda::std::abs;
  using cuda::std::norm;
  using cuda::std::sin;
  using cuda::std::cos;
  using cuda::std::tan;
  using cuda::std::asin;
  using cuda::std::acos;
  using cuda::std::atan;
  using cuda::std::sinh;
  using cuda::std::cosh;
  using cuda::std::tanh;
  using cuda::std::asinh;
  using cuda::std::acosh;
  using cuda::std::atanh;
  using cuda::std::pow;

  // Helper function for no derivative
  template <typename T>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto none(T v) {
    assert(false);
    return v;
  }
  
  template <typename T>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto zero(T v) {
    return 0;
  }

  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dsin(T x) { 
    return cos(x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dcos(T x) { 
    return -sin(x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dtan(T x) { 
	  T c = cos(x); 
    return T(1)/(c*c); 
  }

  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dasin(T x) { 
    return T(1)/(T(1) - x*x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dacos(T x) { 
    return T(1)/(T(1) - x*x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T datan(T x) { 
    return T(1)/(T(1) + x*x); 
  }

  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dsinh(T x) { 
    return (exp(x) + exp(-x))/T(2);
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dcosh(T x) { 
    return (exp(x) + exp(-x))/T(2);  
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dtanh(T x) { 
    T t = tanh(x); 
    return T(1)-t*t; 
  }

  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dasinh(T x) { 
    return T(1)/sqrt(x*x+T(1));
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dacosh(T x) { 
    return T(1)/sqrt(x*x-T(1));
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T datanh(T x) { 
    return T(1)/(1 - x*x);
  }
  
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dsqrt(T x) { 
    return T(0.5)/sqrt(x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dlog(T x) { 
    return T(1)/x; 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dlog2(T x) { 
    return T(1)/(log(2)*x); 
  }
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dlog10(T x) { 
    return T(1)/(log(10)*x); 
  }
  
  template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T dabs(T x) { 
    return x / abs(x);
  }

  template <typename T1, typename T2> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto dproduct(T1 v1, T2 v2, T1 d1, T2 d2) { 
    // product rule
    return v1*d2 + v2*d1;
  }
  
  template <typename T1, typename T2> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto dquotient(T1 v1, T2 v2, T1 d1, T2 d2) { 
    // quotient rule
    return (d1*v2 - d2*v1)/(v2*v2);
  }
  
  template <typename T1, typename T2> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto dadd([[maybe_unused]] T1 v1, [[maybe_unused]] T2 v2, T1 d1, T2 d2) { 
    // sum rule
    return d1 + d2;
  }
  
  template <typename T1, typename T2> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto dsub([[maybe_unused]] T1 v1, [[maybe_unused]] T2 v2, T1 d1, T2 d2) { 
    // difference rule
    return d1 - d2;
  }
  
  template <typename T1, typename T2> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto dpow(T1 v1, T2 v2, [[maybe_unused]] T1 d1, [[maybe_unused]] T2 d2) {
#if 0
    if constexpr (std::is_same<float, T1>::value) {
      printf("dpow: v1: %f, v2: %f, d1: %f, d2: %f\n", (float)v1, (float)v2, (float)d1, (float)d2);
    }
#endif
    // derived from generalized power rule and chain rule
    // https://www.youtube.com/watch?v=SUxcFxM65Ho
    if(d2 != T2(0) )  // this branch arises when exponent is a constant
      return pow(v1,v2) * ( v2 * d1 / v1 + d2 * log(v1));
    else
      return  pow(v1,v2) * ( v2 * d1 / v1);
  }
}  

} // end namespace detail
} // end namespace matx
