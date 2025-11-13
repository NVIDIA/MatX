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
#include <cuda/std/utility>
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


/**
 * @brief Determine if a type is NoShape
 * 
 * @tparam T Type to test
 */
template <class T>
concept is_noshape = cuda::std::is_same_v<detail::NoShape, T>;

// Legacy variable for backwards compatibility
template <class T>
inline constexpr bool is_noshape_v = cuda::std::is_same_v<detail::NoShape, T>;



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

/**
 * @brief Determine if a type is a MatX tie type
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_mtie_c = requires {
  typename remove_cvref_t<T>::mtie_type;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_mtie()
{
  return requires { typename remove_cvref_t<T>::mtie_type; };
}


/**
 * @brief Determine if a type is a MatX operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_op_c = requires {
  typename remove_cvref_t<T>::matxop;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_op()
{
  return requires { typename remove_cvref_t<T>::matxop; };
}

/**
 * @brief Determine if a type is a MatX tensor set operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_tensor_set_op = requires {
  typename remove_cvref_t<T>::tensor_type::tensor_view;
};

/**
 * @brief Determine if a type is a MatX transform set operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_transform_set_op = requires {
  typename remove_cvref_t<T>::op_type::matx_transform_op;
};

/**
 * @brief Determine if a type is a MatX transform operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_transform_op_c = requires {
  typename remove_cvref_t<T>::matx_transform_op;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_transform_op()
{
  return requires { typename remove_cvref_t<T>::matx_transform_op; };
}

/**
 * @brief Determine if a type has can_alias trait
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept has_can_alias_c = requires {
  typename remove_cvref_t<T>::can_alias;
};

/**
 * @brief Determine if operator can alias
 * 
 * Returns true if the type is a transform operator and has the can_alias trait set
 * 
 * @tparam T Type to test
 */
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool can_alias()
{
  return is_matx_transform_op<T>() && (requires { typename remove_cvref_t<T>::can_alias; });
}

/**
 * @brief Determine if a type has op_type member
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept has_matx_op_type = requires {
  typename T::op_type;
};



/**
 * @brief Determine if a type is a MatX set operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_set_op_c = requires {
  typename remove_cvref_t<T>::matx_setop;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_set_op()
{
  return requires { typename remove_cvref_t<T>::matx_setop; };
}



/**
 * @brief Determine if a type is a left hand side operator
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_matx_op_lvalue_c = requires {
  typename T::matxoplvalue;
};

// Legacy function for backwards compatibility
template <typename T> 
constexpr __MATX_HOST__ __MATX_DEVICE__ bool is_matx_op_lvalue()
{
  return requires { typename T::matxoplvalue; };
}

/**
 * @brief Determine if a type is a MatX tensor_t
 * 
 * @tparam T Type to test
 */
template< class T >
concept is_tensor_t = requires {
  typename remove_cvref_t<T>::tensor_t_type;
};

// Legacy variable for backwards compatibility
template< class T >
inline constexpr bool is_tensor_t_v = requires { typename remove_cvref_t<T>::tensor_t_type; };

/**
 * @brief Determine if a type is a MatX tensor_impl_t
 * 
 * @tparam T Type to test
 */
template< class T >
concept is_tensor_impl = requires {
  typename remove_cvref_t<T>::tensor_impl;
};

// Legacy variable for backwards compatibility
template< class T >
inline constexpr bool is_tensor_impl_v = requires { typename remove_cvref_t<T>::tensor_impl; };

/**
 * @brief Determine if a type is a MatX tensor
 * 
 * @tparam T Type to test
 */
template< class T >
concept is_tensor = requires {
  typename remove_cvref_t<T>::tensor_view;
};

// Primary variable template
template< class T >
inline constexpr bool is_tensor_v = requires { typename remove_cvref_t<T>::tensor_view; };

// Legacy alias for backwards compatibility
template< class T >
inline constexpr bool is_tensor_view_v = is_tensor_v<T>;

/**
 * @brief Determine if a type is a cuda::std::tuple
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_tuple_c = requires {
  typename cuda::std::tuple_size<T>::type;
  requires cuda::std::tuple_size<T>::value >= 0;
};

// Legacy struct for backwards compatibility
template <typename> struct is_tuple: cuda::std::false_type {};
template <typename ...T> struct is_tuple<cuda::std::tuple<T...>>: cuda::std::true_type {};

template <typename T>
inline constexpr bool is_settable_xform_v = (requires { typename remove_cvref_t<T>::matx_setop; }) && 
                                             (requires { typename remove_cvref_t<T>::op_type::matx_transform_op; });




/**
 * @brief Determine if a type is a MatX reduction
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_reduction = requires {
  typename T::matx_reduce;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_reduction_v = requires { typename T::matx_reduce; };

/**
 * @brief Determine if a type is a MatX index reduction type
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_index_reduction = requires {
  typename T::matx_reduce_index;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_index_reduction_v = requires { typename T::matx_reduce_index; };

/**
 * @brief Determine if a type is not allowed to use CUB for reductions
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_no_cub_reduction = requires {
  typename T::matx_no_cub_reduce;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_no_cub_reduction_v = requires { typename T::matx_no_cub_reduce; };




/**
 * @brief Determine if a type is a cuda::std::complex variant
 * 
 * @tparam T Type to test
 */
template <class T>
concept is_cuda_complex = requires {
  typename remove_cvref_t<T>::value_type;
  requires cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<typename remove_cvref_t<T>::value_type>>;
};

// Legacy variable for backwards compatibility
template <class T>
inline constexpr bool is_cuda_complex_v = requires {
  typename remove_cvref_t<T>::value_type;
  requires cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<typename remove_cvref_t<T>::value_type>>;
};


/**
 * @brief Determine if a type is a device executor
 * 
 * @tparam T Type to test
 */
template <typename T> 
concept is_cuda_executor = requires {
  typename remove_cvref_t<T>::cuda_executor;
};

// Legacy variable for backwards compatibility
template <typename T> 
inline constexpr bool is_cuda_executor_v = requires { typename remove_cvref_t<T>::cuda_executor; };

/**
 * @brief Determine if a type is a CUDA executor but NOT a JIT CUDA executor
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_cuda_non_jit_executor = requires { typename remove_cvref_t<T>::cuda_executor; }
                                   && !(requires { typename remove_cvref_t<T>::jit_cuda_executor; });

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_cuda_non_jit_executor_v = requires { typename remove_cvref_t<T>::cuda_executor; }
                                                   && !(requires { typename remove_cvref_t<T>::jit_cuda_executor; });

/**
 * @brief Determine if a type is a CUDA JIT executor
 *
 * @tparam T Type to test
 */
template <typename T>
concept is_cuda_jit_executor = requires { typename remove_cvref_t<T>::jit_cuda_executor; };

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_cuda_jit_executor_v = requires { typename remove_cvref_t<T>::jit_cuda_executor; };

/**
 * @brief Determine if a type is a complex type (any type supported)
 * 
 * @tparam T Type to test
 */
template <class T> 
concept is_complex = cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<float>> ||
                     cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<double>> ||
                     cuda::std::is_same_v<remove_cvref_t<T>, matxFp16Complex> ||
                     cuda::std::is_same_v<remove_cvref_t<T>, matxBf16Complex>;

// Legacy variable for backwards compatibility
template <class T> 
inline constexpr bool is_complex_v = cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<float>> ||
                                     cuda::std::is_same_v<remove_cvref_t<T>, cuda::std::complex<double>> ||
                                     cuda::std::is_same_v<remove_cvref_t<T>, matxFp16Complex> ||
                                     cuda::std::is_same_v<remove_cvref_t<T>, matxBf16Complex>;

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
  template <typename T>
  concept has_jit_storage_type = requires {
    typename T::JIT_Storage;
  };

  template <typename T>
  struct inner_storage_t {
    using type = void;
  };

  // Specialization: if T has a member type 'JIT_Storage', use it
  template <has_jit_storage_type T>
  struct inner_storage_t<T> {
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
  template <typename T>
  concept has_to_jit_storage = requires(T t) {
    { t.ToJITStorage() };
  };

  template <typename T>
  inline constexpr bool has_to_jit_storage_v = has_to_jit_storage<T>;
}


/**
 * @brief Get the inner value_type of the container
 * @tparam T Type to test
 */
template <typename T>
struct inner_op_type_t {
  using type = T;
};

template <is_complex T>
struct inner_op_type_t<T> { 
  using type = typename T::value_type;
};


/**
 * @brief Determine if a type is a BF16 type
 * 
 * @tparam T Type to test
 */
template <class T> 
concept is_bf16_type = cuda::std::is_same_v<T, matxBf16Complex> ||
                       cuda::std::is_same_v<T, matxBf16>;

// Legacy variable for backwards compatibility
template <class T> 
inline constexpr bool is_bf16_type_v = cuda::std::is_same_v<T, matxBf16Complex> ||
                                       cuda::std::is_same_v<T, matxBf16>;

/**
 * @brief Determine if a type is an FP16 type
 * 
 * @tparam T Type to test
 */
template <class T> 
concept is_fp16_type = cuda::std::is_same_v<T, matxFp16Complex> ||
                       cuda::std::is_same_v<T, matxFp16>;

// Legacy variable for backwards compatibility
template <class T> 
inline constexpr bool is_fp16_type_v = cuda::std::is_same_v<T, matxFp16Complex> ||
                                       cuda::std::is_same_v<T, matxFp16>;

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

/**
 * @brief Determine if a type is a MatX shape type
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_shape = requires {
  typename remove_cvref_t<T>::matx_shape;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_shape_v = requires { typename remove_cvref_t<T>::matx_shape; };
 

/**
 * @brief Determine if a type is a complex half precision type
 * 
 * @tparam T Type to test
 */
template <class T>
concept is_complex_half = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16Complex> ||
                          cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16Complex>;

// Legacy variable for backwards compatibility
template <class T>
inline constexpr bool is_complex_half_v = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16Complex> ||
                                          cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16Complex>;

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

/**
 * @brief Determine if a type is a MatX half precision wrapper (either matxFp16 or matxBf16)
 * 
 * @tparam T Type to test
 */
template <class T>
concept is_matx_half = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16> ||
                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16>;

// Legacy variable for backwards compatibility
template <class T>
inline constexpr bool is_matx_half_v = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16> ||
                                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16>;

/**
 * @brief Determine if a type is half precision (either __half or __nv_bfloat16)
 * 
 * @tparam T Type to test
 */
template <class T> 
concept is_half = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, __half> ||
                  cuda::std::is_same_v<cuda::std::remove_cv_t<T>, __nv_bfloat16>;

// Legacy variable for backwards compatibility
template <class T> 
inline constexpr bool is_half_v = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, __half> ||
                                  cuda::std::is_same_v<cuda::std::remove_cv_t<T>, __nv_bfloat16>;

/**
 * @brief Determine if a type is a MatX custom type (half precision wrappers)
 * 
 * @tparam T Type to test
 */
template <class T>
concept is_matx_type = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16> ||
                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16> ||
                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16Complex> ||
                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16Complex>;

// Legacy variable for backwards compatibility
template <class T>
inline constexpr bool is_matx_type_v = cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16> ||
                                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16> ||
                                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxFp16Complex> ||
                                       cuda::std::is_same_v<cuda::std::remove_cv_t<T>, matxBf16Complex>;

namespace detail {
template <typename T>
struct extract_value_type_impl {
  using value_type = T;
};

template <is_matx_op_c T>
struct extract_value_type_impl<T> {
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

/**
 * @brief Determine if a type is a MatX descriptor
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_descriptor = requires {
  typename remove_cvref_t<T>::matx_descriptor;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_descriptor_v = requires { typename remove_cvref_t<T>::matx_descriptor; };

/**
 * @brief Determine if a type is a MatX static descriptor
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_static_descriptor = requires {
  typename remove_cvref_t<T>::matx_static_descriptor;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_matx_static_descriptor_v = requires { typename remove_cvref_t<T>::matx_static_descriptor; };


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



template <typename T> 
struct complex_from_scalar {
  using type = T;
};

template <typename T> 
  requires (cuda::std::is_same_v<float, T> || cuda::std::is_same_v<double, T>)
struct complex_from_scalar<T> {
  using type = cuda::std::complex<T>;
};

template <typename T> 
  requires (is_matx_half<T> || is_half<T>)
struct complex_from_scalar<T> {
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
    cuda::std::get<Ints>(cuda::std::forward<Tuple>(tuple))...);
}

template <typename... T>
  requires ((is_tensor<T>) && ...)
constexpr __MATX_HOST__ __MATX_DEVICE__ bool TensorTypesMatch() {
  using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<T...>>;
  return ((cuda::std::is_same_v<typename first_type::value_type, typename T::value_type>) && ...);
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
template <typename T> 
struct base_type {
  using type = T;
};

template <is_tensor_t T> 
struct base_type<T> {
  using type = tensor_impl_t<typename T::value_type, T::Rank(), typename T::desc_type, typename T::data_type>;
};

template <typename T> using base_type_t = typename base_type<typename remove_cvref<T>::type>::type;

}

/**
 * @brief Determine if a type is a MatX sparse data type
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_sparse_data = requires {
  typename remove_cvref_t<T>::sparse_data;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_sparse_data_v = requires { typename remove_cvref_t<T>::sparse_data; };
/**
 * @brief Determine if a type is a MatX sparse tensor type
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_sparse_tensor = requires {
  typename remove_cvref_t<T>::sparse_tensor;
};

// Legacy variable for backwards compatibility
template <typename T>
inline constexpr bool is_sparse_tensor_v = requires { typename remove_cvref_t<T>::sparse_tensor; };

namespace detail {
// Helpers for extracting types in the aliases
  template <typename> struct is_std_tuple: cuda::std::false_type {};
#ifndef __CUDACC_RTC__
  template <typename ...T> struct is_std_tuple<std::tuple<T...>>: cuda::std::true_type {};
#endif
  template <typename ...T> struct is_std_tuple<cuda::std::tuple<T...>>: cuda::std::true_type {};
}

/**
 * @brief Determine if a type is std::array or cuda::std::array
 * 
 * @tparam T Type to test
 */
template<typename T>
concept is_std_array_c = requires {
  typename remove_cvref_t<T>::value_type;
  requires requires { remove_cvref_t<T>::size(); };
  requires cuda::std::tuple_size<remove_cvref_t<T>>::value >= 0;
};

namespace detail {
  template<typename T> struct is_std_array : cuda::std::false_type {};
  template<typename T, size_t N> struct is_std_array<cuda::std::array<T, N>> : cuda::std::true_type {};
#ifndef __CUDACC_RTC__  
  template<typename T, size_t N> struct is_std_array<std::array<T, N>> : cuda::std::true_type {};
#endif
  template <typename T> inline constexpr bool is_std_array_v = is_std_array_c<T>;

/**
 * @brief Determine if a type is a MatX JIT class
 * 
 * @tparam T Type to test
 */
template <typename T>
concept is_matx_jit_class = is_matx_op_c<T> || requires {
  typename remove_cvref_t<T>::emits_jit_str;
};

// Legacy struct for backwards compatibility
template <typename T, typename = void>
struct is_matx_jit_class_impl : cuda::std::bool_constant<is_matx_op_c<T>> {};

template <typename T>
struct is_matx_jit_class_impl<T, cuda::std::void_t<typename remove_cvref_t<T>::emits_jit_str>>
    : cuda::std::true_type {};

template <typename T>
inline constexpr bool is_matx_jit_class_v = (requires { typename remove_cvref_t<T>::matxop; }) || 
                                            (requires { typename remove_cvref_t<T>::emits_jit_str; });
  
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

