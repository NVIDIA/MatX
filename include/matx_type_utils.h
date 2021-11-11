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

#include "cuda_fp16.h"
#include "matx.h"
#include <any>
#include <complex>
#include <cublas_v2.h>
#include <type_traits>

/**
 * Defines type traits for host and device compilers. This file should be includable by
 * the host compiler, so no device code should be present
 */

namespace matx {

template <typename T, typename = void>
struct is_matx_op_impl : std::false_type {
};

template <typename T>
struct is_matx_op_impl<T, std::void_t<typename T::matxop>> : std::true_type {
};

template <typename T> constexpr bool is_matx_op()
{
  return is_matx_op_impl<T>::value;
}

template <typename T, typename = void> struct is_tensor_view : std::false_type {
};

template <typename T>
struct is_tensor_view<T, std::void_t<typename T::tensor_view>>
    : std::true_type {
};

template <typename T> constexpr bool is_tensor_view_t()
{
  return is_tensor_view<T>::value;
}

template <typename T, typename = void>
struct is_matx_reduction_impl : std::false_type {
};
template <typename T>
struct is_matx_reduction_impl<T, std::void_t<typename T::matx_reduce>>
    : std::true_type {
};
template <typename T>
inline constexpr bool is_matx_reduction_v = is_matx_reduction_impl<T>::value;

template <class T> struct is_cuda_complex : std::false_type {
};
template <class T>
struct is_cuda_complex<cuda::std::complex<T>> : std::true_type {
};
template <class T>
inline constexpr bool is_cuda_complex_v = is_cuda_complex<T>::value;

template <typename T> struct is_complex : std::false_type {
};
template <> struct is_complex<cuda::std::complex<float>> : std::true_type {
};
template <> struct is_complex<cuda::std::complex<double>> : std::true_type {
};
template <> struct is_complex<std::complex<float>> : std::true_type {
};
template <> struct is_complex<std::complex<double>> : std::true_type {
};
template <> struct is_complex<matxFp16Complex> : std::true_type {
};
template <> struct is_complex<matxBf16Complex> : std::true_type {
};
template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
struct is_complex_half
    : std::integral_constant<
          bool, std::is_same_v<matxFp16Complex, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16Complex, std::remove_cv_t<T>>> {
};
template <class T>
inline constexpr bool is_complex_half_v = is_complex_half<T>::value;

template <typename T> constexpr inline bool IsHalfType()
{
  return std::is_same_v<T, matxFp16> || std::is_same_v<T, matxBf16>;
}

template <typename T>
struct is_matx_half
    : std::integral_constant<
          bool, std::is_same_v<matxFp16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16, std::remove_cv_t<T>>> {
};
template <class T>
inline constexpr bool is_matx_half_v = is_matx_half<T>::value;

template <typename T>
struct is_half
    : std::integral_constant<
          bool, std::is_same_v<__half, std::remove_cv_t<T>> ||
                    std::is_same_v<__nv_bfloat16, std::remove_cv_t<T>>> {
};
template <class T> inline constexpr bool is_half_v = is_half<T>::value;

template <typename T>
struct is_matx_type
    : std::integral_constant<
          bool, std::is_same_v<matxFp16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxFp16Complex, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16Complex, std::remove_cv_t<T>>> {
};
template <class T>
inline constexpr bool is_matx_type_v = is_matx_type<T>::value;

// Type traits to help with the lack of short-circuit template logic. Numpy
// doesn't support bfloat16 at all, we just use fp32 for the numpy side
template <class T> struct identity {
  using type = typename std::conditional_t<IsHalfType<T>(), float, T>;
};
template <class C>
struct complex_type_of
    : identity<std::complex<std::conditional_t<is_complex_half_v<C>, float,
                                               typename C::value_type>>> {
};

template <class C>
using matx_convert_complex_type =
    typename std::conditional_t<!is_complex_v<C>, identity<C>,
                                complex_type_of<C>>::type;

template <typename T>
using promote_half_t = typename std::conditional_t<is_half_v<T>, float, T>;

template <class T, class = void> struct value_type {
  using type = T;
};
template <class T> struct value_type<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};
template <class T> using value_type_t = typename value_type<T>::type;

template <typename T> using value_promote_t = promote_half_t<value_type_t<T>>;


// Helpers for extracting types in the aliases
template <typename T, typename = void> struct extract_scalar_type_impl {
  using scalar_type = T;
};

template <typename T>
struct extract_scalar_type_impl<T, std::void_t<typename T::scalar_type>> {
  using scalar_type = typename T::scalar_type;
};

template <typename T>
using extract_scalar_type_t = typename extract_scalar_type_impl<T>::scalar_type;



// Supported MatX data types. This enum helps translate types into integers for
// hashing purposes
typedef enum {
  // MATX_TYPE_COMPLEX_FP16, // Not supported until libcu++ supports it
  MATX_TYPE_COMPLEX_FP32,
  MATX_TYPE_COMPLEX_FP64,
  MATX_TYPE_FP16,
  MATX_TYPE_BF16,
  MATX_TYPE_COMPLEX_FP16,
  MATX_TYPE_COMPLEX_BF16,
  MATX_TYPE_FP32,
  MATX_TYPE_FP64,
  MATX_TYPE_INT8,
  MATX_TYPE_INT16,
  MATX_TYPE_INT32,
  MATX_TYPE_INT64,
  MATX_TYPE_UINT8,
  MATX_TYPE_UINT16,
  MATX_TYPE_UINT32,
  MATX_TYPE_UINT64,

  MATX_TYPE_INVALID // Sentinel
} MatXDataType_t;

template <typename T> constexpr MatXDataType_t TypeToInt()
{
  if constexpr (std::is_same_v<T, cuda::std::complex<float>>)
    return MATX_TYPE_COMPLEX_FP32;
  if constexpr (std::is_same_v<T, cuda::std::complex<double>>)
    return MATX_TYPE_COMPLEX_FP64;
  if constexpr (std::is_same_v<T, matxFp16>)
    return MATX_TYPE_FP16;
  if constexpr (std::is_same_v<T, matxBf16>)
    return MATX_TYPE_BF16;
  if constexpr (std::is_same_v<T, matxFp16Complex>)
    return MATX_TYPE_COMPLEX_FP16;
  if constexpr (std::is_same_v<T, matxBf16Complex>)
    return MATX_TYPE_COMPLEX_BF16;
  if constexpr (std::is_same_v<T, float>)
    return MATX_TYPE_FP32;
  if constexpr (std::is_same_v<T, double>)
    return MATX_TYPE_FP64;
  if constexpr (std::is_same_v<T, int8_t>)
    return MATX_TYPE_INT8;
  if constexpr (std::is_same_v<T, int16_t>)
    return MATX_TYPE_INT16;
  if constexpr (std::is_same_v<T, int32_t>)
    return MATX_TYPE_INT32;
  if constexpr (std::is_same_v<T, int64_t>)
    return MATX_TYPE_INT64;
  if constexpr (std::is_same_v<T, uint8_t>)
    return MATX_TYPE_UINT8;
  if constexpr (std::is_same_v<T, uint16_t>)
    return MATX_TYPE_UINT16;
  if constexpr (std::is_same_v<T, uint32_t>)
    return MATX_TYPE_UINT32;
  if constexpr (std::is_same_v<T, uint64_t>)
    return MATX_TYPE_UINT64;

  return MATX_TYPE_INVALID;
}

template <MatXDataType_t IntType> struct IntToType {
};
template <> struct IntToType<MATX_TYPE_COMPLEX_FP32> {
  using value_type = cuda::std::complex<float>;
};
template <> struct IntToType<MATX_TYPE_COMPLEX_FP64> {
  using value_type = cuda::std::complex<double>;
};
template <> struct IntToType<MATX_TYPE_FP16> {
  using value_type = matxFp16;
};
template <> struct IntToType<MATX_TYPE_BF16> {
  using value_type = matxBf16;
};
template <> struct IntToType<MATX_TYPE_COMPLEX_FP16> {
  using value_type = matxFp16Complex;
};
template <> struct IntToType<MATX_TYPE_COMPLEX_BF16> {
  using value_type = matxBf16Complex;
};
template <> struct IntToType<MATX_TYPE_FP32> {
  using value_type = float;
};
template <> struct IntToType<MATX_TYPE_FP64> {
  using value_type = double;
};
template <> struct IntToType<MATX_TYPE_INT8> {
  using value_type = int8_t;
};
template <> struct IntToType<MATX_TYPE_INT16> {
  using value_type = int16_t;
};
template <> struct IntToType<MATX_TYPE_INT32> {
  using value_type = int32_t;
};
template <> struct IntToType<MATX_TYPE_INT64> {
  using value_type = int64_t;
};
template <> struct IntToType<MATX_TYPE_UINT8> {
  using value_type = uint8_t;
};
template <> struct IntToType<MATX_TYPE_UINT16> {
  using value_type = uint16_t;
};
template <> struct IntToType<MATX_TYPE_UINT32> {
  using value_type = uint32_t;
};
template <> struct IntToType<MATX_TYPE_UINT64> {
  using value_type = uint64_t;
};


template <typename T> constexpr cudaDataType_t MatXTypeToCudaType()
{
  if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    return CUDA_C_32F;
  }
  if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    return CUDA_C_64F;
  }
  if constexpr (std::is_same_v<T, int8_t>) {
    return CUDA_R_8I;
  }
  if constexpr (std::is_same_v<T, float>) {
    return CUDA_R_32F;
  }
  if constexpr (std::is_same_v<T, double>) {
    return CUDA_R_64F;
  }
  if constexpr (std::is_same_v<T, matxFp16>) {
    return CUDA_R_16F;
  }
  if constexpr (std::is_same_v<T, matxBf16>) {
    return CUDA_R_16BF;
  }
  if constexpr (std::is_same_v<T, matxFp16Complex>) {
    return CUDA_C_16F;
  }
  if constexpr (std::is_same_v<T, matxBf16Complex>) {
    return CUDA_C_16BF;
  }

  return CUDA_C_32F;
}

template <typename T> constexpr cublasComputeType_t MatXTypeToCudaComputeType()
{
  if constexpr (std::is_same_v<T, cuda::std::complex<float>> ||
                std::is_same_v<T, float> || is_matx_half_v<T> ||
                std::is_same_v<T, matxFp16Complex> ||
                std::is_same_v<T, matxBf16Complex>) {
    return CUBLAS_COMPUTE_32F;
  }
  if constexpr (std::is_same_v<T, cuda::std::complex<double>> ||
                std::is_same_v<T, double>) {
    return CUBLAS_COMPUTE_64F;
  }

  return CUBLAS_COMPUTE_32F;
}



} // end namespace matx
