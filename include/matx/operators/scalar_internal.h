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

// This file contains scalar_internal_* functions that are used by the scalar operator macros.
// It's separated so it can be both included normally and stringified for JIT compilation.

namespace matx {
namespace detail {


// Helper function to apply a callable binary operator onto two inputs. There are many compile-time
// branches in here because we need to handle both scalar and vector inputs on both sides.
template <typename BinOpFunc, typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto BinVecFunc(const BinOpFunc &func, const T1 &v1, const T2 &v2) {
  if constexpr (is_vector_v<T1> && is_vector_v<T2>) {
    static_assert(T1::width == T2::width, "Vector sizes must match");
    detail::Vector<decltype(func(v1.data[0], v2.data[0])), T1::width> res;
    MATX_LOOP_UNROLL
    for (size_t i = 0; i < T1::width; i++) {
      res.data[i] = func(v1.data[i], v2.data[i]);
    }
    return res;
  }
  else if constexpr (is_vector_v<T1> && !is_vector_v<T2>) {
    detail::Vector<decltype(func(v1.data[0], v2)), T1::width> res;
    MATX_LOOP_UNROLL
    for (size_t i = 0; i < T1::width; i++) {
      res.data[i] = func(v1.data[i], v2);
    }
    return res;
  }
  else if constexpr (!is_vector_v<T1> && is_vector_v<T2>) {
    detail::Vector<decltype(func(v1, v2.data[0])), T2::width> res;
    MATX_LOOP_UNROLL
    for (size_t i = 0; i < T2::width; i++) {
      res.data[i] = func(v1, v2.data[i]);
    }
    return res;
  }
  else {
    return func(v1, v2);
  }
}

template <typename UnaryOpFunc, typename T1>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto UnaryVecFunc(const UnaryOpFunc &func, const T1 &v1) {
  if constexpr (is_vector_v<T1>) {
    detail::Vector<decltype(func(v1.data[0])), T1::width> res;
    MATX_LOOP_UNROLL
    for (size_t i = 0; i < T1::width; i++) {
      res.data[i] = func(v1.data[i]);
    }
    return res;
  }
  else {
    return func(v1);
  }
}  

// Unary scalar_internal functions (custom implementations)

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_rsqrt(T v1) {
  if constexpr (is_matx_type_v<T>){
    return rsqrt(v1);
  }
  else {
    #ifdef __CUDACC__
      return ::rsqrt(v1);
    #else
      return static_cast<T>(1.0 / sqrt(v1));
    #endif
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_csqrt(T v1) {
  return sqrt(static_cast<cuda::std::complex<T>>(v1));
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_conj(T v1) {
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::conj(v1);
  }
  else if constexpr (is_complex_half_v<T>) {
    return matx::conj(v1);
  }
  else {
    return v1;
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_sin(T v1) {
  if constexpr (is_matx_type_v<T>) {
    return matx::sin(v1);
  }
  else {
    return cuda::std::sin(v1);
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_cos(T v1) {
  if constexpr (is_matx_type_v<T>) {
    return matx::cos(v1);
  }
  else {
    return cuda::std::cos(v1);
  }
}

template <typename T>
  requires (cuda::std::is_floating_point_v<T> || is_matx_half<T>)
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_expj(T v1) {
  if constexpr (is_matx_half_v<T>) {
    return matxHalfComplex<T>{scalar_internal_cos(v1), scalar_internal_sin(v1)};
  }
  else {
    if constexpr (cuda::std::is_same_v<T, double>) {
      double sinx, cosx;
      sincos(v1, &sinx, &cosx);
      return cuda::std::complex<T>{cosx, sinx};
    } else if constexpr (cuda::std::is_same_v<T, float>) {
      float sinx, cosx;
      sincosf(v1, &sinx, &cosx);
      return cuda::std::complex<T>{cosx, sinx};
    } else {
      return cuda::std::complex<T>{scalar_internal_cos(v1), scalar_internal_sin(v1)};
    }
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_abs2(T v1) {
  if constexpr (is_complex_v<T>) {
    return v1.real() * v1.real() + v1.imag() * v1.imag();
  }
  else {
    return v1 * v1;
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_normcdf(T v1) {
  return normcdf(v1);
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_real(T v1) {
  if constexpr (is_complex_v<T>) {
    return v1.real();
  }
  else {
    return v1;
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_imag(T v1) {
  if constexpr (is_complex_v<T>) {
    return v1.imag();
  }
  else {
    return v1;
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_angle(T v1) {
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::atan2(v1.imag(), v1.real());
  }
  else {
    return atan2(v1.imag(), v1.real());
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_subneg(T v1) {
  return -v1;
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_not(T v1) {
  return !v1;
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_isnan(T v1) {
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
      return false;
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    return cuda::std::isnan(static_cast<typename castType::value_type>(v1.real())) || cuda::std::isnan(static_cast<typename castType::value_type>(v1.imag()));
  } else {
    return cuda::std::isnan(static_cast<castType>(v1));
  }
}

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_isinf(T v1) {
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
    return false;
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    return cuda::std::isinf(static_cast<typename castType::value_type>(v1.real())) || cuda::std::isinf(static_cast<typename castType::value_type>(v1.imag()));
  } else {
    return cuda::std::isinf(static_cast<castType>(v1));
  }
}

// Binary scalar_internal functions (custom implementations)

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_fmod(T1 v1, T2 v2) {
  if constexpr (is_matx_half_v<T1> || is_matx_half_v<T2>) {
    return fmod(v1, v2);
  } else {
    return cuda::std::fmod(v1, v2);
  }
}

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_atan2(T1 v1, T2 v2) {
  if constexpr (is_matx_half_v<T1> || is_matx_half_v<T2>) {
    return atan2(v1, v2);
  }
  else {
    return cuda::std::atan2(v1, v2);
  }
}

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_max(T1 v1, T2 v2) {
  return cuda::std::max(v1, v2);
}

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_min(T1 v1, T2 v2) {
  return cuda::std::min(v1, v2);
}

} // namespace detail
} // namespace matx

