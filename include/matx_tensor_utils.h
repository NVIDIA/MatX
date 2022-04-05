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

namespace matx
{
  template <typename Op>
  size_t TotalSize(const Op &op) {
    if constexpr (is_tensor_view_v<Op>) {
      return static_cast<size_t>(op.TotalSize());
    }
    else {
      size_t total = 1;
      for (int i = 0; i < op.Rank(); i++) {
        total *= op.Size(i);
      }

      return total;
    }
  }

namespace detail {

  /**
   * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs
   * 
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
      return 0;

    // work around for compiler bug/warning
    if constexpr (!is_matx_op<M>())
      return 0;
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
        if (dim > diff)
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

}  
}