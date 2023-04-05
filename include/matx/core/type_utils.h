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

#include <any>
#include <memory>
#include <complex>
#include <cublas_v2.h>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <type_traits>

#include "cuda_fp16.h"
#include "matx/core/half.h"
#include "matx/core/half_complex.h"
#include "matx/executors/device.h"

/**
 * Defines type traits for host and device compilers. This file should be includable by
 * the host compiler, so no device code should be present
 */


namespace matx {

enum {
  matxNoRank = -1
};

enum class MemoryLayout {
  MEMORY_LAYOUT_ROW_MAJOR,
  MEMORY_LAYOUT_COL_MAJOR,
};


/**
 * @brief Removes cv and reference qualifiers on a type
 * 
 * @tparam T Type to remove qualifiers
 */
template< class T >
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>; ///< Type after removal
};  

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T, int RANK, typename Desc> class tensor_impl_t;
template <typename T, int RANK, typename Storage, typename Desc> class tensor_t;

namespace detail {
template <typename T, typename = void>
struct is_matx_op_impl : std::false_type {
};

template <typename T>
struct is_matx_op_impl<T, std::void_t<typename T::matxop>> : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_matx_op()
{
  return detail::is_matx_op_impl<T>::value;
}

namespace detail {
template <typename T, typename = void>
struct is_matx_op_lvalue_impl : std::false_type {
};

template <typename T>
struct is_matx_op_lvalue_impl<T, std::void_t<typename T::matxoplvalue>> : std::true_type {
};
}

/**
 * @brief Determine if a type is a left hand side operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr bool is_matx_op_lvalue()
{
  return detail::is_matx_op_lvalue_impl<T>::value;
}

namespace detail {
template <typename T, typename = void> struct is_tensor_view : std::false_type {
};

template <typename T>
struct is_tensor_view<T, std::void_t<typename T::tensor_view>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX tensor view type
 * 
 * @tparam T Type to test
 */
template< class T >
inline constexpr bool is_tensor_view_v = detail::is_tensor_view<typename remove_cvref<T>::type>::value;

namespace detail {
template <typename T> struct is_executor : std::false_type {};
template <> struct is_executor<cudaExecutor> : std::true_type {};
template <> struct is_executor<SingleThreadHostExecutor> : std::true_type {};
}

/**
 * @brief Determine if a type is a MatX executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
constexpr bool is_executor_t()
{
  return detail::is_executor<typename remove_cvref<T>::type>::value;
}


namespace detail {
template<typename T> struct is_device_executor : std::false_type {};
template<> struct is_device_executor<matx::cudaExecutor> : std::true_type {};
}

/**
 * @brief Determine if a type is a device executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
inline constexpr bool is_device_executor_v = detail::is_device_executor<typename remove_cvref<T>::type>::value;


namespace detail {
template <typename T, typename = void>
struct is_matx_reduction_impl : std::false_type {
};
template <typename T>
struct is_matx_reduction_impl<T, std::void_t<typename T::matx_reduce>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX reduction
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_reduction_v = detail::is_matx_reduction_impl<T>::value;

namespace detail {
template <typename T, typename = void>
struct is_matx_idx_reduction_impl : std::false_type {
};
template <typename T>
struct is_matx_idx_reduction_impl<T, std::void_t<typename T::matx_reduce_index>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX index reduction type
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_index_reduction_v = detail::is_matx_idx_reduction_impl<T>::value;

namespace detail {
template <typename T, typename = void>
struct is_matx_no_cub_reduction_impl : std::false_type {
};
template <typename T>
struct is_matx_no_cub_reduction_impl<T, std::void_t<typename T::matx_no_cub_reduce>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is not allowed to use CUB for reductions
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_no_cub_reduction_v = detail::is_matx_no_cub_reduction_impl<T>::value;

namespace detail {
template<typename T> struct is_smart_ptr : std::false_type {};
template<typename T> struct is_smart_ptr<std::shared_ptr<T>> : std::true_type {};
template<typename T> struct is_smart_ptr<std::unique_ptr<T>> : std::true_type {};
}

/**
 * @brief Determine if a type is a smart pointer (unique or shared)
 * 
 * @tparam T Type to test
 */
template <typename T> inline constexpr bool is_smart_ptr_v = detail::is_smart_ptr<T>::value;

namespace detail {
template <class T> struct is_cuda_complex : std::false_type {
};
template <class T>
struct is_cuda_complex<cuda::std::complex<T>> : std::true_type {
};
}

/**
 * @brief Determine if a type is a cuda::std::complex variant
 * 
 * @tparam T Type to test
 */
template <class T>
inline constexpr bool is_cuda_complex_v = detail::is_cuda_complex<remove_cvref_t<T>>::value;




namespace detail {
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
}

/**
 * @brief Determine if a type is a complex type (any type supported)
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_complex_v = detail::is_complex<T>::value;


/**
 * @brief Get the inner value_type of the container
 * @tparam T Type to test
 */
template <typename T, typename = void>
struct inner_op_type_t {
  using type = T;
};

template <typename T>
struct inner_op_type_t<T, typename std::enable_if_t<is_complex_v<T>>> { 
  using type = typename T::value_type;
};


namespace detail {
template <typename T> struct is_bf16_type : std::false_type {};
template <> struct is_bf16_type<matxBf16Complex> : std::true_type {};
template <> struct is_bf16_type<matxBf16> : std::true_type {};
}

/**
 * @brief Determine if a type is a BF16 type
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_bf16_type_v = detail::is_bf16_type<T>::value;

namespace detail {
template <typename T> struct is_fp16_type : std::false_type {};
template <> struct is_fp16_type<matxFp16Complex> : std::true_type {};
template <> struct is_fp16_type<matxFp16> : std::true_type {};

}

/**
 * @brief Determine if a type is an FF16 type
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_fp16_type_v = detail::is_fp16_type<T>::value;


namespace detail {
template <typename T, typename = void>
struct is_matx_shape : std::false_type {
};
template <typename T>
struct is_matx_shape<T, std::void_t<typename T::matx_shape>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX shape type
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_shape_v = detail::is_matx_shape<typename remove_cvref<T>::type>::value;


namespace detail {
template <typename T, typename = void>
struct is_matx_storage : std::false_type {
};
template <typename T>
struct is_matx_storage<T, std::void_t<typename T::matx_storage>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX storage type
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_storage_v = detail::is_matx_storage<typename remove_cvref<T>::type>::value;

namespace detail {
template <typename T, typename = void>
struct is_matx_storage_container : std::false_type {
};
template <typename T>
struct is_matx_storage_container<T, std::void_t<typename T::matx_storage_container>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX storage container
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_storage_container_v = detail::is_matx_storage_container<typename remove_cvref<T>::type>::value;


namespace detail {
template <typename T, typename = void>
struct is_matx_descriptor : std::false_type {
};
template <typename T>
struct is_matx_descriptor<T, std::void_t<typename T::matx_descriptor>>
    : std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX descriptor
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_descriptor_v = detail::is_matx_descriptor<typename remove_cvref<T>::type>::value;


namespace detail {
template <typename T>
struct is_complex_half
    : std::integral_constant<
          bool, std::is_same_v<matxFp16Complex, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16Complex, std::remove_cv_t<T>>> {
};
}

/**
 * @brief Determine if a type is a MatX half precision wrapper (either matxFp16 or matxBf16)
 * 
 * @tparam T Type to test
 */
template <class T>
inline constexpr bool is_complex_half_v = detail::is_complex_half<T>::value;

/**
 * @brief Tests if a type is a half precision floating point
 * 
 * @tparam T Type to test
 * @return True if half precision floating point
 */
template <typename T> constexpr inline bool IsHalfType()
{
  return std::is_same_v<T, matxFp16> || std::is_same_v<T, matxBf16>;
}

namespace detail {
template <typename T>
struct is_matx_half
    : std::integral_constant<
          bool, std::is_same_v<matxFp16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16, std::remove_cv_t<T>>> {
};
}
/**
 * @brief Determine if a type is a MatX half precision wrapper (either matxFp16 or matxBf16)
 * 
 * @tparam T Type to test
 */
template <class T>
inline constexpr bool is_matx_half_v = detail::is_matx_half<T>::value;

namespace detail {
template <typename T>
struct is_half
    : std::integral_constant<
          bool, std::is_same_v<__half, std::remove_cv_t<T>> ||
                    std::is_same_v<__nv_bfloat16, std::remove_cv_t<T>>> {
};
}
/**
 * @brief Determine if a type is half precision (either __half or __nv_bfloat16)
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_half_v = detail::is_half<T>::value;

namespace detail {
template <typename T>
struct is_matx_type
    : std::integral_constant<
          bool, std::is_same_v<matxFp16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16, std::remove_cv_t<T>> ||
                    std::is_same_v<matxFp16Complex, std::remove_cv_t<T>> ||
                    std::is_same_v<matxBf16Complex, std::remove_cv_t<T>>> {
};
}

/**
 * @brief Determine if a type is a MatX custom type (half precision wrappers)
 * 
 * @tparam T Type to test
 */
template <class T>
inline constexpr bool is_matx_type_v = detail::is_matx_type<T>::value;

namespace detail {
template <typename T, typename = void> struct extract_scalar_type_impl {
  using scalar_type = T;
};

template <typename T>
struct extract_scalar_type_impl<T, std::void_t<typename T::scalar_type>> {
  using scalar_type = typename T::scalar_type;
};
}

/**
 * @brief Extract the scalar_type type
 * 
 * @tparam T Type to extract from
 */
template <typename T>
using extract_scalar_type_t = typename detail::extract_scalar_type_impl<T>::scalar_type;

/**
 * @brief Promote half precision floating point value to fp32, or leave untouched if not half
 * 
 * @tparam T Type to convert
 */
template <typename T>
using promote_half_t = typename std::conditional_t<is_half_v<T>, float, T>;



namespace detail {
  
template <typename T> 
struct convert_matx_type {
  using type = T;
};

template <> 
struct convert_matx_type<matxFp16> {
  using type = __half;
};

template <> 
struct convert_matx_type<matxBf16> {
  using type = __nv_bfloat16;
};

template <typename T> 
using convert_matx_type_t = typename convert_matx_type<T>::type;

template <typename T> 
struct convert_half_to_matx_half {
  using type = T;
};

template <> 
struct convert_half_to_matx_half<__half> {
  using type = matxFp16;
};

template <> 
struct convert_half_to_matx_half<__nv_bfloat16> {
  using type = matxBf16;
};

template <typename T> 
using convert_half_to_matx_half_t = typename convert_half_to_matx_half<T>::type;



template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N>
    to_array_impl(T (&a)[N], std::index_sequence<I...>)
{
    return { {a[I]...} };
}

template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
{
    return to_array_impl(a, std::make_index_sequence<N>{});
}

template <typename T, int RANK, typename Storage, typename Desc> class tensor_t;
template <typename T, int RANK, typename Desc> class tensor_impl_t;
// Traits for casting down to impl tensor conditionally
template <typename T, typename = void> 
struct base_type {
  using type = T;
};

template <typename T> 
struct base_type<T, typename std::enable_if_t<is_tensor_view_v<T>>> {
  using type = tensor_impl_t<typename T::scalar_type, T::Rank(), typename T::desc_type>;
};

template <typename T> using base_type_t = typename base_type<typename remove_cvref<T>::type>::type;

template <typename T, typename = void> 
struct complex_from_scalar {
  using type = T;
};

template <typename T> 
struct complex_from_scalar<T, typename std::enable_if_t<std::is_same_v<float, T> || std::is_same_v<double, T>>> {
  using type = cuda::std::complex<T>;
};

template <typename T> 
struct complex_from_scalar<T, typename std::enable_if_t<is_matx_half_v<T> || is_half_v<T> > > {
  using type = matxHalfComplex<typename convert_half_to_matx_half<T>::type>;
};

template <typename T> using complex_from_scalar_t = typename complex_from_scalar<typename remove_cvref<T>::type>::type;


template <typename T, typename = void> 
struct exec_type {
  using type = T;
};

template <typename T> 
struct exec_type<T, typename std::enable_if_t<std::is_same_v<T, int>>> {
  using type = cudaExecutor;
};

template <typename T> using exec_type_t = typename exec_type<typename remove_cvref<T>::type>::type;

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


template <class T, class = void> struct value_type {
  using type = T;
};
template <class T> struct value_type<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};
template <class T> using value_type_t = typename value_type<T>::type;

template <typename T> using value_promote_t = promote_half_t<value_type_t<T>>;


// Helpers for extracting types in the aliases


template <typename> struct is_std_tuple: std::false_type {};
template <typename ...T> struct is_std_tuple<std::tuple<T...>>: std::true_type {};

template<typename T> struct is_std_array : std::false_type {};
template<typename T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};
template <typename T> inline constexpr bool is_std_array_v = detail::is_std_array<remove_cvref_t<T>>::value;



// Get the n-th element from a parameter pack
template <int I, class... Ts>
__MATX_DEVICE__ __MATX_HOST__ decltype(auto) pp_get(Ts&&... ts) {
  return std::get<I>(std::forward_as_tuple(ts...));
}

template <std::size_t ... Is>
constexpr auto index_sequence_rev(std::index_sequence<Is...> const &)
   -> decltype( std::index_sequence<sizeof...(Is) -1U - Is...>{} );

template <std::size_t N>
using make_index_sequence_rev
   = decltype(index_sequence_rev(std::make_index_sequence<N>{}));


// Taken from Ramond Chen's blog entries on tuple tricks
template<std::size_t N, typename Seq> struct offset_sequence;

template<std::size_t N, std::size_t... Ints>
struct offset_sequence<N, std::index_sequence<Ints...>>
{
 using type = std::index_sequence<Ints + N...>;
};
template<std::size_t N, typename Seq>
using offset_sequence_t = typename offset_sequence<N, Seq>::type;

template<typename Tuple, std::size_t... Ints>
__MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto select_tuple(Tuple&& tuple, std::index_sequence<Ints...>)
{
 return cuda::std::tuple<cuda::std::tuple_element_t<Ints, Tuple>...>(
    cuda::std::get<Ints>(std::forward<Tuple>(tuple))...);
}

template <typename... T, std::enable_if_t<((is_tensor_view_v<T>) && ...), bool> = true>
constexpr bool TensorTypesMatch() {
  using first_type = std::tuple_element_t<0, std::tuple<T...>>;
  return ((std::is_same_v<typename first_type::scalar_type, typename T::scalar_type>) && ...);
}

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
} // end namespace detail

} // end namespace matx
