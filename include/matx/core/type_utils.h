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

#include "matx/core/operator_options.h"
#include "matx/core/type_utils_both.h"

#include <memory>
#include <cublas_v2.h>
#include <cusparse.h>

#include "cuda_fp16.h"
#include "matx/executors/jit_cuda.h"
#include "matx/executors/cuda.h"
#include "matx/executors/host.h"
#include "matx/core/half.h"
#include "matx/core/half_complex.h"


/**
 * Defines type traits specific to the host compiler. This file should be includable by
 * the host compiler, so no device code should be present
 */


namespace matx {
  

/**
 * @brief Determine if a type is a MatX executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_executor = requires {
  typename remove_cvref_t<T>::matx_executor;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr bool is_executor_t()
{
  return requires { typename remove_cvref_t<T>::matx_executor; };
}

/**
 * @brief Determine if a type is a CUDA JIT executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_jit_cuda_executor = std::is_same_v<remove_cvref_t<T>, CUDAJITExecutor>;

// Legacy function for backwards compatibility
template <typename T> 
constexpr bool is_jit_cuda_executor_t()
{
  return std::is_same_v<remove_cvref_t<T>, CUDAJITExecutor>;
}

/**
 * @brief Determine if a type is a host executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_host_executor = requires {
  typename remove_cvref_t<T>::host_executor;
};

// Legacy variable for backwards compatibility
template <typename T> 
inline constexpr bool is_host_executor_v = requires { typename remove_cvref_t<T>::host_executor; };

/**
 * @brief Determine if a type is a select threads host executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_select_threads_host_executor = std::is_same_v<remove_cvref_t<T>, matx::SelectThreadsHostExecutor>;

// Legacy variable for backwards compatibility
template <typename T> 
inline constexpr bool is_select_threads_host_executor_v = std::is_same_v<remove_cvref_t<T>, matx::SelectThreadsHostExecutor>;

/**
 * @brief Determine if a type is a std::complex variant
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_std_complex = requires {
  requires std::is_same_v<remove_cvref_t<T>, std::complex<typename remove_cvref_t<T>::value_type>>;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_std_complex_v = requires { 
  requires std::is_same_v<remove_cvref_t<T>, std::complex<typename remove_cvref_t<T>::value_type>>;
};


/**
 * @brief Determine if a type is a smart pointer (unique or shared)
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_smart_ptr = requires {
  requires std::is_same_v<T, std::shared_ptr<typename T::element_type>> ||
           std::is_same_v<T, std::unique_ptr<typename T::element_type>>;
};

// Legacy variable for backwards compatibility
template <typename T> 
inline constexpr bool is_smart_ptr_v = requires {
  requires std::is_same_v<T, std::shared_ptr<typename T::element_type>> ||
           std::is_same_v<T, std::unique_ptr<typename T::element_type>>;
};

/**
 * @brief Determine if a type is a MatX storage type
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_storage = requires {
  typename remove_cvref_t<T>::matx_storage;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_storage_v = requires { typename remove_cvref_t<T>::matx_storage; };

/**
 * @brief Determine if a type is a MatX storage container
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_storage_container = requires {
  typename remove_cvref_t<T>::matx_storage_container;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_storage_container_v = requires { typename remove_cvref_t<T>::matx_storage_container; };




/**
 * @brief Determine if a type defines `using index_cmp_op = bool;`
 *
 * @tparam T Type to test
 */
template <typename T>
concept has_index_cmp_op = requires {
  typename remove_cvref_t<T>::index_cmp_op;
  requires std::is_same_v<typename remove_cvref_t<T>::index_cmp_op, bool>;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool has_index_cmp_op_v = requires {
  typename remove_cvref_t<T>::index_cmp_op;
  requires std::is_same_v<typename remove_cvref_t<T>::index_cmp_op, bool>;
};



namespace detail {

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
  if constexpr (std::is_same_v<T, int>) {
    return CUDA_R_32I;
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

template <typename T>
constexpr cusparseIndexType_t MatXTypeToCuSparseIndexType() {
  if constexpr (std::is_same_v<T, uint16_t>) {
    return CUSPARSE_INDEX_16U;
  }
  if constexpr (std::is_same_v<T, int32_t>) {
    return CUSPARSE_INDEX_32I;
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    return CUSPARSE_INDEX_64I;
  }
  if constexpr (std::is_same_v<T, index_t>) {
    return CUSPARSE_INDEX_64I;
  }
  else { // Should not happen
    return CUSPARSE_INDEX_32I;
  }
}


} // end namespace detail

} // end namespace matx