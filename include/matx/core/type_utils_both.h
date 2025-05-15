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

#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include "matx/core/half.h"
#include "matx/core/half_complex.h"
#include "matx/core/operator_options.h"
#ifndef __CUDACC_RTC__
#include <complex>
#endif



/**
 * Defines type traits that work on both host and device compilers.
 */

namespace matx {

enum {
  matxNoRank = -1
};

enum class MemoryLayout {
  MEMORY_LAYOUT_ROW_MAJOR,
  MEMORY_LAYOUT_COL_MAJOR,
};


namespace detail {

template <typename T>
struct is_noshape : cuda::std::integral_constant<bool, cuda::std::is_same_v<NoShape, T>> {};
};


/**
 * @brief Determine if a type is a MatX half precision wrapper (either matxFp16 or matxBf16)
 * 
 * @tparam T Type to test
 */
template <class T>
inline constexpr bool is_noshape_v = detail::is_noshape<T>::value;



/**
 * @brief Removes cv and reference qualifiers on a type
 * 
 * @tparam T Type to remove qualifiers
 */
template< class T >
struct remove_cvref {
    using type = cuda::std::remove_cv_t<cuda::std::remove_reference_t<T>>; ///< Type after removal
};  

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T, int RANK, typename Desc, typename Data> class tensor_impl_t;
template <typename T, int RANK, typename Desc> class tensor_t;

namespace detail {
template <typename T, typename = void>
struct is_mtie_impl : cuda::std::false_type {
};

template <typename T>
struct is_mtie_impl<T, cuda::std::void_t<typename T::mtie_type>> : cuda::std::true_type {
};
}

template <typename T> constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_mtie()
{
  return detail::is_mtie_impl<typename remove_cvref<T>::type>::value;
}


namespace detail {
template <typename T, typename = void>
struct is_matx_op_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_op_impl<T, cuda::std::void_t<typename T::matxop>> : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_op()
{
  return detail::is_matx_op_impl<typename remove_cvref<T>::type>::value;
}

namespace detail {
template <typename T, typename = void>
struct is_matx_tensor_set_op_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_tensor_set_op_impl<T, cuda::std::void_t<typename remove_cvref_t<T>::tensor_type::tensor_view>> : cuda::std::true_type {
};
}

namespace detail {
template <typename T, typename = void>
struct is_matx_transform_set_op_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_transform_set_op_impl<T, cuda::std::void_t<typename remove_cvref_t<T>::op_type::matx_transform_op>> : cuda::std::true_type {
};
}

namespace detail {
template <typename T, typename = void>
struct is_matx_transform_op_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_transform_op_impl<T, cuda::std::void_t<typename remove_cvref_t<T>::matx_transform_op>> : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX transform operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_transform_op()
{
  return detail::is_matx_transform_op_impl<typename remove_cvref<T>::type>::value;
}

namespace detail {
template <typename T, typename = void>
struct has_matx_op_type : cuda::std::false_type {
};

template <typename T>
struct has_matx_op_type<T, cuda::std::void_t<typename T::op_type>> : cuda::std::true_type {
};
}



namespace detail {
template <typename T, typename = void>
struct is_matx_set_op_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_set_op_impl<T, cuda::std::void_t<typename T::matx_setop>> : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX set operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_set_op()
{
  return detail::is_matx_set_op_impl<typename remove_cvref<T>::type>::value;
}



namespace detail {
template <typename T, typename = void>
struct is_matx_op_lvalue_impl : cuda::std::false_type {
};

template <typename T>
struct is_matx_op_lvalue_impl<T, cuda::std::void_t<typename T::matxoplvalue>> : cuda::std::true_type {
};
}



/**
 * @brief Determine if a type is a left hand side operator
 * 
 * @tparam T Type to test
 */
template <typename T> constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_op_lvalue()
{
  return detail::is_matx_op_lvalue_impl<T>::value;
}

namespace detail {
template <typename T, typename = void> struct is_tensor_t : cuda::std::false_type {};
template <typename T>
struct is_tensor_t<T, cuda::std::void_t<typename T::tensor_t_type>> : cuda::std::true_type {};
}

/**
 * @brief Determine if a type is a MatX tensor_t
 * 
 * @tparam T Type to test
 */
template< class T >
inline constexpr bool is_tensor_t_v = detail::is_tensor_t<typename remove_cvref<T>::type>::value;

namespace detail {
template <typename T, typename = void> struct is_tensor_impl : cuda::std::false_type {};
template <typename T>
struct is_tensor_impl<T, cuda::std::void_t<typename T::tensor_impl>> : cuda::std::true_type {};
}

/**
 * @brief Determine if a type is a MatX tensor_impl_t
 * 
 * @tparam T Type to test
 */
template< class T >
inline constexpr bool is_tensor_impl_v = detail::is_tensor_impl<typename remove_cvref<T>::type>::value;

namespace detail {
template <typename T, typename = void> struct is_tensor_view : cuda::std::false_type {
};

template <typename T>
struct is_tensor_view<T, cuda::std::void_t<typename T::tensor_view>>
    : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX tensor view type
 * 
 * @tparam T Type to test
 */
template< class T >
inline constexpr bool is_tensor_view_v = detail::is_tensor_view<typename remove_cvref<T>::type>::value;

template <typename> struct is_tuple: cuda::std::false_type {};
template <typename ...T> struct is_tuple<cuda::std::tuple<T...>>: cuda::std::true_type {};

template <typename T>
inline constexpr bool is_settable_xform_v = cuda::std::conjunction_v<detail::is_matx_set_op_impl<T>, 
                                               detail::is_matx_transform_set_op_impl<T>>;
                                               //detail::is_matx_tensor_set_op_impl<T>>; // Can be tuple also




namespace detail {
template <typename T, typename = void>
struct is_matx_reduction_impl : cuda::std::false_type {
};
template <typename T>
struct is_matx_reduction_impl<T, cuda::std::void_t<typename T::matx_reduce>>
    : cuda::std::true_type {
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
struct is_matx_idx_reduction_impl : cuda::std::false_type {
};
template <typename T>
struct is_matx_idx_reduction_impl<T, cuda::std::void_t<typename T::matx_reduce_index>>
    : cuda::std::true_type {
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
struct is_matx_no_cub_reduction_impl : cuda::std::false_type {
};
template <typename T>
struct is_matx_no_cub_reduction_impl<T, cuda::std::void_t<typename T::matx_no_cub_reduce>>
    : cuda::std::true_type {
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
template <class T> struct is_cuda_complex : cuda::std::false_type {
};
template <class T>
struct is_cuda_complex<cuda::std::complex<T>> : cuda::std::true_type {
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
  template <typename T, typename = void> struct is_cuda_executor : cuda::std::false_type {};
  
  template <typename T>
  struct is_cuda_executor<T, cuda::std::void_t<typename T::cuda_executor>> : cuda::std::true_type {};
}

/**
  * @brief Determine if a type is a device executor
  * 
  * @tparam T Type to test
  */
template <typename T> 
inline constexpr bool is_cuda_executor_v = detail::is_cuda_executor<typename remove_cvref<T>::type>::value;


namespace detail {
template <typename T> struct is_complex : cuda::std::false_type {
};
template <> struct is_complex<cuda::std::complex<float>> : cuda::std::true_type {
};
template <> struct is_complex<cuda::std::complex<double>> : cuda::std::true_type {
};
template <> struct is_complex<matxFp16Complex> : cuda::std::true_type {
};
template <> struct is_complex<matxBf16Complex> : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a complex type (any type supported)
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_complex_v = detail::is_complex<remove_cvref_t<T>>::value;

namespace detail {
template <typename T> struct scalar_to_complex {
  using ctype = T;
};
template <> struct scalar_to_complex<float>  {
  using ctype = cuda::std::complex<float>;
};
template <> struct scalar_to_complex<double>  {
  using ctype = cuda::std::complex<double>;
};
template <> struct scalar_to_complex<matxFp16>  {
  using ctype = matxFp16Complex;
};
template <> struct scalar_to_complex<matxBf16>  {
  using ctype = matxBf16Complex;
};
}

// Primary template: fallback to T if no inner_storage member exists
namespace detail {
  template <typename, typename = void>
  struct inner_storage_t {
    using type = void;
  };

  // Specialization: if T has a member type 'inner_storage', use it
  template <typename T>
  struct inner_storage_t<T, cuda::std::void_t<typename T::JIT_Storage>> {
    using type = typename T::JIT_Storage;
  };

  // Helper alias: if T has inner_storage, use it; else use T itself
  template <typename T>
  using inner_storage_or_self_t = cuda::std::conditional_t<
    !cuda::std::is_void_v<typename inner_storage_t<T>::type>,
    typename inner_storage_t<T>::type,
    T
  >;
}

namespace detail {
  // Helper to check if a type has ToJITStorage method
  template <typename, typename = void>
  struct has_to_jit_storage : cuda::std::false_type {};

  template <typename T>
  struct has_to_jit_storage<T, cuda::std::void_t<decltype(cuda::std::declval<T>().ToJITStorage())>> 
    : cuda::std::true_type {};

  template <typename T>
  inline constexpr bool has_to_jit_storage_v = has_to_jit_storage<T>::value;
}


/**
 * @brief Get the inner value_type of the container
 * @tparam T Type to test
 */
template <typename T, typename = void>
struct inner_op_type_t {
  using type = T;
};

template <typename T>
struct inner_op_type_t<T, typename cuda::std::enable_if_t<is_complex_v<T>>> { 
  using type = typename T::value_type;
};


namespace detail {
template <typename T> struct is_bf16_type : cuda::std::false_type {};
template <> struct is_bf16_type<matxBf16Complex> : cuda::std::true_type {};
template <> struct is_bf16_type<matxBf16> : cuda::std::true_type {};
}

/**
 * @brief Determine if a type is a BF16 type
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_bf16_type_v = detail::is_bf16_type<T>::value;

namespace detail {
template <typename T> struct is_fp16_type : cuda::std::false_type {};
template <> struct is_fp16_type<matxFp16Complex> : cuda::std::true_type {};
template <> struct is_fp16_type<matxFp16> : cuda::std::true_type {};

}

/**
 * @brief Determine if a type is an FP16 type
 * 
 * @tparam T Type to test
 */
template <class T> inline constexpr bool is_fp16_type_v = detail::is_fp16_type<T>::value;

/**
 * @brief Determine if the inner type is an FP32 type
 * 
 * @tparam T Type to test
 */
template<typename T>
inline constexpr bool is_fp32_inner_type_v = cuda::std::is_same_v<typename inner_op_type_t<T>::type, float>;

/**
 * @brief Determine if the inner type is an FP64 type
 * 
 * @tparam T Type to test
 */
template<typename T>
inline constexpr bool is_fp64_inner_type_v = cuda::std::is_same_v<typename inner_op_type_t<T>::type, double>;

namespace detail {
template <typename T, typename = void>
struct is_matx_shape : cuda::std::false_type {
};
template <typename T>
struct is_matx_shape<T, cuda::std::void_t<typename T::matx_shape>>
    : cuda::std::true_type {
};
}



namespace detail {
template <typename T>
struct is_complex_half
    : cuda::std::integral_constant<
          bool, cuda::std::is_same_v<matxFp16Complex, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<matxBf16Complex, cuda::std::remove_cv_t<T>>> {
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
template <typename T> constexpr inline __MATX_HOST__ __MATX_DEVICE__ bool IsHalfType()
{
  return cuda::std::is_same_v<T, matxFp16> || cuda::std::is_same_v<T, matxBf16>;
}

namespace detail {
template <typename T>
struct is_matx_half
    : cuda::std::integral_constant<
          bool, cuda::std::is_same_v<matxFp16, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<matxBf16, cuda::std::remove_cv_t<T>>> {
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
    : cuda::std::integral_constant<
          bool, cuda::std::is_same_v<__half, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<__nv_bfloat16, cuda::std::remove_cv_t<T>>> {
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
    : cuda::std::integral_constant<
          bool, cuda::std::is_same_v<matxFp16, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<matxBf16, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<matxFp16Complex, cuda::std::remove_cv_t<T>> ||
                    cuda::std::is_same_v<matxBf16Complex, cuda::std::remove_cv_t<T>>> {
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
template <typename T, typename = void> struct extract_value_type_impl {
  using value_type = T;
};

template <typename T>
struct extract_value_type_impl<T, typename cuda::std::enable_if_t<is_matx_op<T>()>> {
  using value_type = typename T::value_type;
};
}

/**
 * @brief Extract the value_type type
 * 
 * @tparam T Type to extract from
 */
template <typename T>
using extract_value_type_t = typename detail::extract_value_type_impl<cuda::std::remove_cv_t<T>>::value_type;

/**
 * @brief Promote half precision floating point value to fp32, or leave untouched if not half
 * 
 * @tparam T Type to convert
 */
template <typename T>
using promote_half_t = typename cuda::std::conditional_t<is_half_v<T> || is_matx_half_v<T>, float, T>;

namespace detail {
template <typename T, typename = void>
struct is_matx_descriptor : cuda::std::false_type {
};
template <typename T>
struct is_matx_descriptor<T, cuda::std::void_t<typename T::matx_descriptor>>
    : cuda::std::true_type {
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
template <typename T, typename = void>
struct is_matx_static_descriptor : cuda::std::false_type {
};
template <typename T>
struct is_matx_static_descriptor<T, cuda::std::void_t<typename T::matx_static_descriptor>>
    : cuda::std::true_type {
};
}

/**
 * @brief Determine if a type is a MatX static descriptor
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_matx_static_descriptor_v = detail::is_matx_static_descriptor<typename remove_cvref<T>::type>::value;


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



template <class T, cuda::std::size_t N, cuda::std::size_t... I>
constexpr __MATX_HOST__ __MATX_DEVICE__ cuda::std::array<cuda::std::remove_cv_t<T>, N>
    to_array_impl(T (&a)[N], cuda::std::index_sequence<I...>)
{
    return { {a[I]...} };
}

template <class T, cuda::std::size_t N>
constexpr __MATX_HOST__ __MATX_DEVICE__ cuda::std::array<cuda::std::remove_cv_t<T>, N> to_array(T (&a)[N])
{
    return to_array_impl(a, cuda::std::make_index_sequence<N>{});
}



template <typename T, typename = void> 
struct complex_from_scalar {
  using type = T;
};

template <typename T> 
struct complex_from_scalar<T, typename cuda::std::enable_if_t<cuda::std::is_same_v<float, T> || cuda::std::is_same_v<double, T>>> {
  using type = cuda::std::complex<T>;
};

template <typename T> 
struct complex_from_scalar<T, typename cuda::std::enable_if_t<is_matx_half_v<T> || is_half_v<T> > > {
  using type = matxHalfComplex<typename convert_half_to_matx_half<T>::type>;
};

template <typename T> using complex_from_scalar_t = typename complex_from_scalar<typename remove_cvref<T>::type>::type;



// Type traits to help with the lack of short-circuit template logic. Numpy
// doesn't support bfloat16 at all, we just use fp32 for the numpy side
template <class T> struct identity {
  using type = typename cuda::std::conditional_t<IsHalfType<T>(), float, T>;
};


template <class C>
struct cuda_complex_type_of
    : identity<cuda::std::complex<cuda::std::conditional_t<is_complex_half_v<C>, float,
                                               typename C::value_type>>> {
};

template <class C>
using matx_convert_cuda_complex_type =
    typename cuda::std::conditional_t<!is_complex_v<C>, identity<C>,
                                cuda_complex_type_of<C>>::type;                                

#ifndef __CUDACC_RTC__
template <class C>
struct complex_type_of
    : identity<std::complex<cuda::std::conditional_t<is_complex_half_v<C>, float,
                                               typename C::value_type>>> {
};

template <class C>
using matx_convert_complex_type =
    typename cuda::std::conditional_t<!is_complex_v<C>, identity<C>,
                                complex_type_of<C>>::type;
#endif

template <class T, class = void> struct value_type {
  using type = T;
};
template <class T> struct value_type<T, cuda::std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};
template <class T> using value_type_t = typename value_type<T>::type;

template <typename T> using value_promote_t = promote_half_t<value_type_t<T>>;



// Get the n-th element from a parameter pack
template <int I, class... Ts>
__MATX_DEVICE__ __MATX_HOST__ decltype(auto) pp_get(Ts&&... ts) {
  return cuda::std::get<I>(cuda::std::forward_as_tuple(ts...));
}

template <cuda::std::size_t ... Is>
constexpr __MATX_HOST__ __MATX_DEVICE__ auto index_sequence_rev(cuda::std::index_sequence<Is...> const &)
   -> decltype( cuda::std::index_sequence<sizeof...(Is) -1U - Is...>{} );

template <cuda::std::size_t N>
using make_index_sequence_rev
   = decltype(index_sequence_rev(cuda::std::make_index_sequence<N>{}));


// Taken from Ramond Chen's blog entries on tuple tricks
template<cuda::std::size_t N, typename Seq> struct offset_sequence;

template<cuda::std::size_t N, cuda::std::size_t... Ints>
struct offset_sequence<N, cuda::std::index_sequence<Ints...>>
{
 using type = cuda::std::index_sequence<Ints + N...>;
};
template<cuda::std::size_t N, typename Seq>
using offset_sequence_t = typename offset_sequence<N, Seq>::type;

template<typename Tuple, cuda::std::size_t... Ints>
__MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto select_tuple(Tuple&& tuple, cuda::std::index_sequence<Ints...>)
{
 return cuda::std::tuple<cuda::std::tuple_element_t<Ints, Tuple>...>(
    cuda::std::get<Ints>(std::forward<Tuple>(tuple))...);
}

template <typename... T, cuda::std::enable_if_t<((is_tensor_view_v<T>) && ...), bool> = true>
constexpr __MATX_HOST__ __MATX_DEVICE__ bool TensorTypesMatch() {
  using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<T...>>;
  return ((std::is_same_v<typename first_type::value_type, typename T::value_type>) && ...);
}

struct no_permute_t{};

struct fft_t{};
struct ifft_t{};

struct no_size_t{}; // Used to create 0D tensors

template <typename T1, typename T2>
struct permute_rank {
  static const int rank = T1::Rank() - cuda::std::tuple_size_v<T2>;
};

template <typename T1>
struct permute_rank<T1, no_permute_t> {
  static const int rank = 0;
};  



template <typename T, int RANK, typename Storage, typename Desc> class tensor_t;
template <typename T, int RANK, typename Desc, typename Data> class tensor_impl_t;
// Traits for casting down to impl tensor conditionally
template <typename T, typename = void> 
struct base_type {
  using type = T;
};

template <typename T> 
struct base_type<T, typename cuda::std::enable_if_t<is_tensor_t_v<T>>> {
  using type = tensor_impl_t<typename T::value_type, T::Rank(), typename T::desc_type, typename T::data_type>;
};

template <typename T> using base_type_t = typename base_type<typename remove_cvref<T>::type>::type;

}

namespace detail {
template <typename T, typename = void>
struct is_sparse_data : cuda::std::false_type {
};
template <typename T>
struct is_sparse_data<T, cuda::std::void_t<typename T::sparse_data>>
    : cuda::std::true_type {
};
}


/**
 * @brief Determine if a type is a MatX sparse data type
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_sparse_data_v = detail::is_sparse_data<typename remove_cvref<T>::type>::value;
namespace detail {
template <typename T, typename = void>
struct is_sparse_tensor : cuda::std::false_type {
};
template <typename T>
struct is_sparse_tensor<T, cuda::std::void_t<typename T::sparse_tensor>>
    : cuda::std::true_type {
};
}
/**
 * @brief Determine if a type is a MatX sparse tensor type
 * 
 * @tparam T Type to test
 */
template <typename T>
inline constexpr bool is_sparse_tensor_v = detail::is_sparse_tensor<typename remove_cvref<T>::type>::value;

namespace detail {
// Helpers for extracting types in the aliases
  template <typename> struct is_std_tuple: cuda::std::false_type {};
#ifndef __CUDACC_RTC__
  template <typename ...T> struct is_std_tuple<std::tuple<T...>>: cuda::std::true_type {};
#endif
  template <typename ...T> struct is_std_tuple<cuda::std::tuple<T...>>: cuda::std::true_type {};

  template<typename T> struct is_std_array : cuda::std::false_type {};
  template<typename T, size_t N> struct is_std_array<cuda::std::array<T, N>> : cuda::std::true_type {};
#ifndef __CUDACC_RTC__  
  template<typename T, size_t N> struct is_std_array<std::array<T, N>> : cuda::std::true_type {};
#endif
  template <typename T> inline constexpr bool is_std_array_v = detail::is_std_array<remove_cvref_t<T>>::value;

template <typename T, typename = void>
struct is_matx_jit_class : cuda::std::bool_constant<is_matx_op<T>()> {};

template <typename T>
struct is_matx_jit_class<T, cuda::std::void_t<typename remove_cvref_t<T>::emits_jit_str>>
    : cuda::std::true_type {};

template <typename T>
inline constexpr bool is_matx_jit_class_v = is_matx_jit_class<remove_cvref_t<T>>::value;
  
}  

namespace detail {
  template <typename T> 
  struct inner_precision {
    using type = T;
  };

  template <> 
  struct inner_precision<cuda::std::complex<float>> {
    using type = float;
  };  

  template <> 
  struct inner_precision<cuda::std::complex<double>> {
    using type = double;
  };  
  
  template <> 
  struct inner_precision<matxFp16> {
    using type = __half;
  };
  
  template <> 
  struct inner_precision<matxBf16> {
    using type = __nv_bfloat16;
  };
  
  template <> 
  struct inner_precision<matxFp16Complex> {
    using type = __half;
  };
  
  template <> 
  struct inner_precision<matxBf16Complex> {
    using type = __nv_bfloat16;
  };  
}

namespace detail {

  template <typename T, size_t N>
  struct VecTypeSelector {
    // Helper to make static_assert dependent on template parameters
    static constexpr bool always_false = sizeof(T) == 0;
  
    static_assert(N >= 1 && N <= 4, "VecTypeSelector only supports vector sizes 1, 2, 3, and 4");
    static_assert(always_false, "VecTypeSelector: No specialization available for this type and size combination. Check the documentation for supported types.");
  };
  
  template <> struct VecTypeSelector<float, 1> { using type = float1; };
  template <> struct VecTypeSelector<float, 2> { using type = float2; };
  template <> struct VecTypeSelector<float, 3> { using type = float3; };
  template <> struct VecTypeSelector<float, 4> { using type = float4; };
  
  template <> struct VecTypeSelector<double, 1> { using type = double1; };
  template <> struct VecTypeSelector<double, 2> { using type = double2; };
  template <> struct VecTypeSelector<double, 3> { using type = double3; };
  #if CUDART_VERSION >= 13000
  template <> struct VecTypeSelector<double, 4> { using type = double4_32a; };
  #else
  template <> struct VecTypeSelector<double, 4> { using type = double4; };
  #endif
  
  template <> struct VecTypeSelector<char, 1> { using type = char1; };
  template <> struct VecTypeSelector<char, 2> { using type = char2; };
  template <> struct VecTypeSelector<char, 3> { using type = char3; };
  template <> struct VecTypeSelector<char, 4> { using type = char4; };
  
  template <> struct VecTypeSelector<unsigned char, 1> { using type = uchar1; };
  template <> struct VecTypeSelector<unsigned char, 2> { using type = uchar2; };
  template <> struct VecTypeSelector<unsigned char, 3> { using type = uchar3; };
  template <> struct VecTypeSelector<unsigned char, 4> { using type = uchar4; };
  
  template <> struct VecTypeSelector<short, 1> { using type = short1; };
  template <> struct VecTypeSelector<short, 2> { using type = short2; };
  template <> struct VecTypeSelector<short, 3> { using type = short3; };
  template <> struct VecTypeSelector<short, 4> { using type = short4; };
  
  template <> struct VecTypeSelector<unsigned short, 1> { using type = ushort1; };
  template <> struct VecTypeSelector<unsigned short, 2> { using type = ushort2; };
  template <> struct VecTypeSelector<unsigned short, 3> { using type = ushort3; };
  template <> struct VecTypeSelector<unsigned short, 4> { using type = ushort4; };
  
  template <> struct VecTypeSelector<int, 1> { using type = int1; };
  template <> struct VecTypeSelector<int, 2> { using type = int2; };
  template <> struct VecTypeSelector<int, 3> { using type = int3; };
  template <> struct VecTypeSelector<int, 4> { using type = int4; };
  
  template <> struct VecTypeSelector<unsigned int, 1> { using type = uint1; };
  template <> struct VecTypeSelector<unsigned int, 2> { using type = uint2; };
  template <> struct VecTypeSelector<unsigned int, 3> { using type = uint3; };
  template <> struct VecTypeSelector<unsigned int, 4> { using type = uint4; };
  
  template <> struct VecTypeSelector<long, 1> { using type = long1; };
  template <> struct VecTypeSelector<long, 2> { using type = long2; };
  template <> struct VecTypeSelector<long, 3> { using type = long3; };
  #if CUDART_VERSION >= 13000
  template <> struct VecTypeSelector<long, 4> { using type = long4_32a; };
  #else
  template <> struct VecTypeSelector<long, 4> { using type = long4; };
  #endif
  
  template <> struct VecTypeSelector<unsigned long, 1> { using type = ulong1; };
  template <> struct VecTypeSelector<unsigned long, 2> { using type = ulong2; };
  template <> struct VecTypeSelector<unsigned long, 3> { using type = ulong3; };
  #if CUDART_VERSION >= 13000
  template <> struct VecTypeSelector<unsigned long, 4> { using type = ulong4_32a; };
  #else
  template <> struct VecTypeSelector<unsigned long, 4> { using type = ulong4; };
  #endif
  
  template <> struct VecTypeSelector<long long, 1> { using type = longlong1; };
  template <> struct VecTypeSelector<long long, 2> { using type = longlong2; };
  template <> struct VecTypeSelector<long long, 3> { using type = longlong3; };
  #if CUDART_VERSION >= 13000
  template <> struct VecTypeSelector<long long, 4> { using type = longlong4_32a; };
  #else
  template <> struct VecTypeSelector<long long, 4> { using type = longlong4; };
  #endif
  
  template <> struct VecTypeSelector<unsigned long long, 1> { using type = ulonglong1; };
  template <> struct VecTypeSelector<unsigned long long, 2> { using type = ulonglong2; };
  template <> struct VecTypeSelector<unsigned long long, 3> { using type = ulonglong3; };
  #if CUDART_VERSION >= 13000
  template <> struct VecTypeSelector<unsigned long long, 4> { using type = ulonglong4_32a; };
  #else
  template <> struct VecTypeSelector<unsigned long long, 4> { using type = ulonglong4; };
  #endif
  
  template <typename... Ts>
  struct AggregateToVec {
    static_assert(sizeof...(Ts) > 0, "AggregateToVec requires at least one type");
  
  private:
    static constexpr bool any_half = ((is_half_v<Ts> || is_matx_half_v<Ts>) || ...);
  
  public:
    using common_type = cuda::std::common_type_t<Ts...>;
  
    static_assert(!cuda::std::is_void_v<common_type>, "Types must have a common type");
    static_assert(!any_half,  "zipvec does not support input operators with half types");
  
    using type = typename VecTypeSelector<common_type, sizeof...(Ts)>::type;
  };
  
  template <typename... Ts>
  using AggregateToVecType = typename AggregateToVec<Ts...>::type;  
}


} // end namespace matx

