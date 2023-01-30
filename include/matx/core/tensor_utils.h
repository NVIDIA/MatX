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

#include <cuda/std/tuple>
#include <functional>

#include "matx/core/nvtx.h"
#include "matx/core/make_tensor.h"

namespace matx
{
    /**
   * @brief Returns Total Size of the Operation
   *
   * @param op Operator
   * @return size_t size of data
   */
  template <typename Op>
  index_t TotalSize(const Op &op) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    
    if constexpr (is_tensor_view_v<Op>) {
      return static_cast<size_t>(op.TotalSize());
    }
    else {
      index_t total = 1;
      for (int i = 0; i < op.Rank(); i++) {
        total *= op.Size(i);
      }

      return total;
    }

    return 0;
  }


  /**
   * @brief finds the size of the largest dim of the tensor
   *core/tensor_utils.h
   * @param op Operator
   * @return size of largest dim
   */
  template <typename Op>
  index_t LargestDimSize(const Op &op) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    index_t maxSize = op.Size(0);

    for (int i = 1; i < op.Rank(); i++)
    {
      maxSize = std::max(op.Size(i), maxSize);
    }

    return maxSize;
  }

namespace detail {

  /**
   * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs
   *
   * @param op Operator
   * @param abs Absolute index
   * @return std::array of indices
   */
  template <typename Op>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto GetIdxFromAbs(const Op &op, index_t abs) {
    using l_stride_type = index_t;
    using l_shape_type = index_t;
    constexpr int RANK = op.Rank();

    std::array<l_shape_type, RANK> indices;

    for (int idx = 0; idx < RANK; idx++) {
      if (idx == RANK-1) {
        indices[RANK-1] = abs;
      }
      else {
        // no std::accumulate on the device
        l_stride_type prod = 1;
        for (int i = idx + 1; i < RANK; i++) {
          prod *= op.Size(i);
        }

        indices[idx] = abs / prod;
        abs -= prod * indices[idx];
      }
    }

    return indices;
  }

  /**
   * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs mapping
   * to a block index. Non-batched dims are removed from the computation
   *
   * @param op Operator
   * @param abs Absolute index
   * @param nb_dims Non-batched dims
   * @return std::array of indices
   */
  template <typename Op>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto BlockToIdx(const Op &op, index_t abs, int nb_dims) {
    using l_stride_type = index_t;
    using l_shape_type = index_t;
    constexpr int RANK = op.Rank();
    std::array<l_shape_type, RANK> indices{0};

    for (int idx = 0; idx < RANK - nb_dims; idx++) {
      if (idx == RANK-nb_dims-1) {
        indices[RANK - nb_dims - 1] = abs;
      }
      else {
        // no std::accumulate on the device
        l_stride_type prod = 1;
        for (int i = idx + 1; i < RANK - nb_dims; i++) {
          prod *= op.Size(i);
        }

        indices[idx] = abs / prod;
        abs -= prod * indices[idx];
      }
    }

    return indices;
  }

  // Work around cuda::std::apply not working
  template <typename Func, typename Tuple, size_t... S>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) apply_impl(Func &&f, Tuple&& tuple, std::index_sequence<S...>)  {
    
    if constexpr (is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value) {
      return cuda::std::invoke(std::forward<Func>(f), std::get<S>(std::forward<Tuple>(tuple))...);
    }
    else {
      return cuda::std::invoke(std::forward<Func>(f), cuda::std::get<S>(std::forward<Tuple>(tuple))...);
    }

    if constexpr (!(is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value)) {
            return cuda::std::invoke(std::forward<Func>(f), cuda::std::get<S>(std::forward<Tuple>(tuple))...);
    }
    else {
      return cuda::std::invoke(std::forward<Func>(f), std::get<S>(std::forward<Tuple>(tuple))...);
    }
  }

  template <class Func, class Tuple>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr decltype(auto) mapply(Func&& f, Tuple&& t)
  {
    if constexpr (is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value) {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
    else {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          std::make_index_sequence<cuda::std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }

    if constexpr (!(is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value)) {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          std::make_index_sequence<cuda::std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
    else {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
  }

  template <class Func, class Tuple>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr decltype(auto) mapply_reverse(Func&& f, Tuple&& t)
  {
    if constexpr (is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value) {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          make_index_sequence_rev<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
    else {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          make_index_sequence_rev<cuda::std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }

    if constexpr (!(is_std_tuple<std::remove_reference_t<Tuple>>::value || is_std_array<std::remove_reference_t<Tuple>>::value)) {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          make_index_sequence_rev<cuda::std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
    else {
      return apply_impl(
          std::forward<Func>(f), std::forward<Tuple>(t),
          make_index_sequence_rev<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
  }

  template <typename T0, typename T1, typename... Tn>
  constexpr auto matx_max(T0 &&t0, T1 &&t1, Tn &&... tn)
  {
    
    if constexpr (sizeof...(tn) == 0) {
        return t0 > t1 ? t0 : t1;
    }
    else {
        return matx_max(matx_max(t0, t1), tn...);
    }

    return t0; // 11.4 compiler has a bug. This is dead code
  }

  template <class T, class M = T>
  __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t get_rank()
  {   
    if constexpr (is_matx_op<M>())
      return T::Rank();
    else
      return -1;

    // work around for compiler bug/warning
    if constexpr (!is_matx_op<M>())
      return -1;
    else
      return T::Rank();
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto get_size([[maybe_unused]] T &a,
                                              [[maybe_unused]] uint32_t dim)
  {
    if constexpr (is_matx_op<M>())
      return a.Size(dim);
    else
      return 1;

    // work around for compiler bug/warning
    if constexpr (!is_matx_op<M>())
      return 1;
    else
      return a.Size(dim);
  }

  template <int RANK, class T, class M = T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
  get_expanded_size([[maybe_unused]] T &a, [[maybe_unused]] uint32_t dim)
  {
    index_t size = 0;
    constexpr int32_t rank = get_rank<T>();

    if constexpr (rank > 0)
    {
      constexpr int32_t diff = RANK - rank;
      if constexpr (diff > 0)
      {
        // auto expansion case,  remap dimension by difference in ranks
        if (dim >= diff)
        {
          size = get_size(a, dim - diff);
        }
      }
      else
      {
        size = get_size(a, dim);
      }
    }

    return size;
  }



  /**
   * @brief Get the matx value object using broadcasting
   *
   * @tparam T type of operator
   * @tparam Is type of indices
   * @param i operator
   * @param indices indices
   * @return Value after broadcasting
   */
  template <class T, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(T &i, Is... indices)
  {
    if constexpr (T::Rank() == sizeof...(Is)) {
      return i(indices...);
    }
    else
    {
      // Construct an integer sequence of the length of the tuple, but only using the last indices
      using seq = offset_sequence_t<sizeof...(Is) - T::Rank(), std::make_index_sequence<T::Rank()>>;
      auto tup = cuda::std::make_tuple(indices...);
      auto sliced_tup = select_tuple(std::forward<decltype(tup)>(tup), seq{});
      return mapply([&](auto... args) {
        return i(args...);
      }, sliced_tup);
    }

    if constexpr (!(T::Rank() == sizeof...(Is))) {
      // Construct an integer sequence of the length of the tuple, but only using the last indices
      using seq = offset_sequence_t<sizeof...(Is) - T::Rank(), std::make_index_sequence<T::Rank()>>;
      auto tup = cuda::std::make_tuple(indices...);
      auto sliced_tup = select_tuple(std::forward<decltype(tup)>(tup), seq{});
      return mapply([&](auto... args) {
        return i(args...);
      }, sliced_tup);
    }
    else
    {
      return i(indices...);
    }
  }


  template <class T, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i, Is... indices)
  {
    if constexpr (is_matx_op<T>())
    {
      return get_matx_value(i, indices...);
    }
    else
    {
      return i;
    }

    if constexpr (!is_matx_op<T>())
    {
      return i;
    }
    else
    {
      return get_matx_value(i, indices...);
    }
  }

  template <typename T> __MATX_INLINE__ std::string to_short_str() {
    if constexpr (!is_complex_v<T>) {
      if constexpr (std::is_same_v<T, bool>)
        return "b";
      if constexpr (std::is_same_v<T, int32_t>)
        return "i32";
      if constexpr (std::is_same_v<T, uint32_t>)
        return "u32";
      if constexpr (std::is_same_v<T, int64_t>)
        return "i64";
      if constexpr (std::is_same_v<T, uint64_t>)
        return "u64";
      if constexpr (std::is_same_v<T, float>)
        return "f32";
      if constexpr (std::is_same_v<T, double>)
        return "f64";
      if constexpr (std::is_same_v<T, matxHalf<__half>>)
        return "f16";
      if constexpr (std::is_same_v<T, matxHalf<__nv_bfloat16>>)
        return "bf16";
      else
        return "x" + std::to_string(sizeof(T)*8);
    }
    else {
      if constexpr (std::is_same_v<typename T::value_type, int32_t>)
        return "i32c";
      if constexpr (std::is_same_v<typename T::value_type, uint32_t>)
        return "u32c";
      if constexpr (std::is_same_v<typename T::value_type, int64_t>)
        return "i64c";
      if constexpr (std::is_same_v<typename T::value_type, uint64_t>)
        return "u64c";
      if constexpr (std::is_same_v<typename T::value_type, float>)
        return "f32c";
      if constexpr (std::is_same_v<typename T::value_type, double>)
        return "f64c";
      if constexpr (std::is_same_v<typename T::value_type, matxHalf<__half>>)
        return "f16";
      if constexpr (std::is_same_v<typename T::value_type, matxHalf<__nv_bfloat16>>)
        return "bf16";
      else
        return "x" + std::to_string(sizeof(typename T::value_type)*8) + "c";
    }
  }

  template <class T> 
  __MATX_INLINE__ __MATX_HOST__  auto get_type_str( [[maybe_unused]] T op) {
     if constexpr (is_matx_op<T>()) {
       return op.str(); 
     } else {
       // This should be a scalar value
       return "S_" + to_short_str<T>();
     }
  }


  // Returns an address of a pointer of type T aligned to new address
  template <typename T>
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T *AlignAddr(uint8_t *addr)
  {
    if (((uint64_t)addr % std::alignment_of_v<T>) != 0) {
      return reinterpret_cast<T *>(
          ((uint64_t)addr + (std::alignment_of_v<T> - 1)) /
          std::alignment_of_v<T> * std::alignment_of_v<T>);
    }

    return reinterpret_cast<T *>(addr);
  }

  template <typename T, typename I, int32_t R>
  void UpdateIndices(const T& op, std::array<I, R> &idx, int res) {
    for (int32_t r = T::Rank() - res - 1; r >= 0; r--) {
      idx[r]++;
      if (idx[r] == op.Size(r)) {
        idx[r] = 0;
      }
      else {
        return;
      }
    }
  }


  /**
   * Print a value
   *
   * Type-agnostic function to print a value to stdout
   *
   * @param val
   */
  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ void PrintVal(const T &val)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    if constexpr (is_complex_v<T>) {
      printf("%.4e%+.4ej ", static_cast<float>(val.real()),
            static_cast<float>(val.imag()));
    }
    else if constexpr (is_matx_half_v<T> || is_half_v<T>) {
      printf("%.4e ", static_cast<float>(val));
    }
    else if constexpr (std::is_floating_point_v<T>) {
      printf("%.4e ", val);
    }
    else if constexpr (std::is_same_v<T, long long int>) {
      printf("%lld ", val);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
      printf("%" PRId64 " ", val);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
      printf("%" PRId32 " ", val);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
      printf("%" PRId16 " ", val);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
      printf("%" PRId8 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
      printf("%" PRIu64 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
      printf("%" PRIu32 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
      printf("%" PRIu16 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
      printf("%" PRIu8 " ", val);
    }
    else if constexpr (std::is_same_v<T, bool>) {
      printf("%d ", val);
    }
  }

  /**
   * convert Type to string
   *
   * function convert a tensor type to a string
   *
   */
  template <typename T> static std::string GetTensorType()
  {
    if constexpr (std::is_same_v<T, bool>)
      return "bool";    
    if constexpr (std::is_same_v<T, int32_t>)
      return "int32_t";
    if constexpr (std::is_same_v<T, uint32_t>)
      return "uint32_t";
    if constexpr (std::is_same_v<T, int64_t>)
      return "int64_t";
    if constexpr (std::is_same_v<T, uint64_t>)
      return "uint64_t";
    if constexpr (std::is_same_v<T, float> )
      return "float";
    if constexpr (std::is_same_v<T, matxFp16>)
      return "float16";
    if constexpr (std::is_same_v<T, matxBf16>)
      return "bfloat16";
    if constexpr (std::is_same_v<T, double>)
      return "double";
    if constexpr (std::is_same_v<T, cuda::std::complex<double>> || std::is_same_v<T, std::complex<double>>) 
      return "complex<double>";
    if constexpr (std::is_same_v<T, cuda::std::complex<float>> || std::is_same_v<T, std::complex<float>>) 
      return "complex<float>";
    if constexpr (std::is_same_v<T, matxFp16Complex>)
      return "complex<float16>";
    if constexpr (std::is_same_v<T, matxBf16Complex>)
      return "complex<bfloat16>";
          
    return "unknown";
  }


  /**
   * Print a tensor
   *
   * Type-agnostic function to print a tensor to stdout
   *
   */
  template <typename Op, typename ... Args>
  __MATX_HOST__ void InternalPrint(const Op &op, Args ...dims)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_STATIC_ASSERT(op.Rank() == sizeof...(Args), "Number of dimensions to print must match tensor rank");
    MATX_STATIC_ASSERT(op.Rank() <= 4, "Printing is only supported on tensors of rank 4 or lower currently");

    if constexpr (sizeof...(Args) == 0) {
      PrintVal(op.operator()());
      printf("\n");
    }
    else if constexpr (sizeof...(Args) == 1) {
      auto& k =detail:: pp_get<0>(dims...);
      for (index_t _k = 0; _k < ((k == 0) ? op.Size(0) : k); _k++) {
        printf("%06lld: ", _k);
        PrintVal(op.operator()(_k));
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 2) {
      auto& k = detail::pp_get<0>(dims...);
      auto& l = detail::pp_get<1>(dims...);
      for (index_t _k = 0; _k < ((k == 0) ? op.Size(0) : k); _k++) {
        for (index_t _l = 0; _l < ((l == 0) ? op.Size(1) : l); _l++) {
          if (_l == 0)
            printf("%06lld: ", _k);

          PrintVal(op.operator()(_k, _l));
        }
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 3) {
      auto& j = detail::pp_get<0>(dims...);
      auto& k = detail::pp_get<1>(dims...);
      auto& l = detail::pp_get<2>(dims...);
      for (index_t _j = 0; _j < ((j == 0) ? op.Size(0) : j); _j++) {
        printf("[%06lld,:,:]\n", _j);
        for (index_t _k = 0; _k < ((k == 0) ? op.Size(1) : k); _k++) {
          for (index_t _l = 0; _l < ((l == 0) ? op.Size(2) : l); _l++) {
            if (_l == 0)
              printf("%06lld: ", _k);

            PrintVal(op.operator()(_j, _k, _l));
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 4) {
      auto& i = detail::pp_get<0>(dims...);
      auto& j = detail::pp_get<1>(dims...);
      auto& k = detail::pp_get<2>(dims...);
      auto& l = detail::pp_get<3>(dims...);
      for (index_t _i = 0; _i < ((i == 0) ? op.Size(0) : i); _i++) {
        for (index_t _j = 0; _j < ((j == 0) ? op.Size(1) : j); _j++) {
          printf("[%06lld,%06lld,:,:]\n", _i, _j);
          for (index_t _k = 0; _k < ((k == 0) ? op.Size(2) : k); _k++) {
            for (index_t _l = 0; _l < ((l == 0) ? op.Size(3) : l); _l++) {
              if (_l == 0)
                printf("%06lld: ", _k);

              PrintVal(op.operator()(_i, _j, _k, _l));
            }
            printf("\n");
          }
          printf("\n");
        }
      }
    }
  }
} // end namespace detail

static constexpr bool PRINT_ON_DEVICE = false;      ///< Print() uses printf on device

/**
 * @brief Print a tensor's values to stdout
 *
 * This is a wrapper utility function to print the type, size and stride of tensor,
 * see PrintData for details of internal tensor printing options
 *
 * @tparam Args Integral argument types
 * @param op input Operator
 * @param dims Number of values to print for each dimension
 */
template <typename Op, typename... Args,
          std::enable_if_t<((std::is_integral_v<Args>)&&...) &&
                                (Op::Rank() == 0 || sizeof...(Args) > 0),
                            bool> = true>
void Print(const Op &op, Args... dims) 
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  // print tensor size info first
  std::string type = (is_tensor_view_v<Op>) ? "Tensor" : "Operator";

  printf("%s{%s} Rank: %d, Sizes:[", type.c_str(), detail::GetTensorType<typename Op::scalar_type>().c_str(), op.Rank());
  
  for (index_t dimIdx = 0; dimIdx < (op.Rank() ); dimIdx++ )
  {
    printf("%lld", op.Size(static_cast<int>(dimIdx)) );
    if( dimIdx < (op.Rank() - 1) )
      printf(", ");
  }
  
  if constexpr (is_tensor_view_v<Op>) 
  {
    printf("], Strides:[");
    if constexpr (Op::Rank() > 0) 
    {
      for (index_t dimIdx = 0; dimIdx < (op.Rank() ); dimIdx++ ) 
      {
        printf("%lld", op.Stride(static_cast<int>(dimIdx)) );
        if( dimIdx < (op.Rank() - 1) )
        {
          printf(",");
        }
      }   
    }
  }

  printf("]\n");
  PrintData(op, dims...);
  
}

/**
 * @brief Print a tensor's values to stdout
 *
 * This is the interal `Print()` takes integral values for each index, and prints that as many values
 * in each dimension as the arguments specify. For example:
 *
 * `a.Print(2, 3, 2);`
 *
 * Will print 2 values of the first, 3 values of the second, and 2 values of the third dimension
 * of a 3D tensor. The number of parameters must match the rank of the tensor. A special value of
 * 0 can be used if the entire tensor should be printed:
 *
 * `a.Print(0, 0, 0);` // Prints the whole tensor
 *
 * For more fine-grained printing, see the over `Print()` overloads.
 *
 * @tparam Args Integral argument types
 * @param op input Operator
 * @param dims Number of values to print for each dimension
 */
template <typename Op, typename... Args,
          std::enable_if_t<((std::is_integral_v<Args>)&&...) &&
                                (Op::Rank() == 0 || sizeof...(Args) > 0),
                            bool> = true>
void PrintData(const Op &op, Args... dims) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
#ifdef __CUDACC__
  if constexpr (is_tensor_view_v<Op>) {

    auto kind = GetPointerKind(op.Data());
    cudaDeviceSynchronize();
    if (HostPrintable(kind)) {
      detail::InternalPrint(op, dims...);
    }
    else if (DevicePrintable(kind) || kind == MATX_INVALID_MEMORY) {
      if constexpr (PRINT_ON_DEVICE) {
        PrintKernel<<<1, 1>>>(op, dims...);
      }
      else {
        auto tmpv = make_tensor<typename Op::scalar_type>(op.Shape());
        (tmpv = op).run();
        PrintData(tmpv, dims...);
      }
    }
  }
  else {
    InternalPrint(op, dims...);
  }
#else
  InternalPrint(op, dims...);
#endif
}

/**
 * @brief Print a tensor's all values to stdout
 *
 * This form of `Print()` is an alias of `Print(0)`, `Print(0, 0)`,
 * `Print(0, 0, 0)` and `Print(0, 0, 0, 0)` for 1D, 2D, 3D and 4D tensor
 * respectively. It passes the proper number of zeros to `Print(...)`
 * automatically according to the rank of this tensor. The user only have to
 * invoke `.Print()` to print the whole tensor, instead of passing zeros
 * manually.
 */
template <typename Op, typename... Args,
          std::enable_if_t<(Op::Rank() > 0 && sizeof...(Args) == 0), bool> = true>
void Print(const Op &op, Args... dims) {
  std::array<int, Op::Rank()> arr = {0};
  auto tp = std::tuple_cat(arr);
  std::apply([&](auto &&...args) { Print(op, args...); }, tp);
}

}
