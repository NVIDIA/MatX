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

#include "matx/core/type_utils_both.h"
#include <cuda/std/__algorithm/copy.h>
#ifndef __CUDACC_RTC__

namespace matx {
  namespace detail {
    template <int N, typename Executor, typename TupleType, typename... Ops>
    void assign_tuple_tensors(const Executor &exec, TupleType &t, Ops... ops)
    {
      if constexpr (N < sizeof...(Ops)) {
        auto in_tup = cuda::std::make_tuple(ops...);
        if (!cuda::std::get<N>(t).isSameView(cuda::std::get<N>(in_tup))) {
          (cuda::std::get<N>(t) = cuda::std::get<N>(in_tup)).run(exec);
          assign_tuple_tensors<N + 1>(exec, t, ops...);
        }
      }
    }
  };  

  constexpr bool RankGTE(int32_t rank1, int32_t rank2) {
    return rank1 >= rank2 || rank1 == matxNoRank;
  }

  constexpr bool RankGT(int32_t rank1, int32_t rank2) {
    return rank1 > rank2 || rank1 == matxNoRank;
  }

  template <typename Op>
  cuda::std::array<index_t, Op::Rank()> Shape(const Op &op) {
    cuda::std::array<index_t, Op::Rank()> shape;
    for (int r = 0; r < Op::Rank(); r++) {
      shape[r] = op.Size(r);
    }

    return shape;
  }
  namespace detail {
    // Used inside of transforms to allocate temporary output
    template <typename TensorType, typename Executor, typename ShapeType>
    __MATX_HOST__ __MATX_INLINE__ void AllocateTempTensor(TensorType &tensor, Executor &&ex, ShapeType &&shape, typename TensorType::value_type **ptr) {

      const auto ttl_size = cuda::std::accumulate(shape.begin(), shape.end(), static_cast<index_t>(1),
                                  cuda::std::multiplies<index_t>()) * sizeof(typename TensorType::value_type);      

      if constexpr (is_cuda_executor_v<Executor>) {
        matxAlloc((void**)ptr, ttl_size, MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
        make_tensor(tensor, *ptr, shape);
      }
      else {
        matxAlloc((void**)ptr, ttl_size, MATX_HOST_MEMORY);
        make_tensor(tensor, *ptr, shape);
      }
    }

    template <typename Op, typename ValidFunc>
    __MATX_INLINE__ auto GetSupportedTensor(const Op &in, const ValidFunc &fn, matxMemorySpace_t space, cudaStream_t stream = 0) {
      if constexpr (is_matx_transform_op<Op>()) {
        // We can assume that if a transform is passed to the input then PreRun has already completed
        // on the transform and we can use the internal pointer
        return make_tensor<typename Op::value_type>(in.Data(), Shape(in));
      }
      else if constexpr (!is_tensor_view_v<Op>) {
        return make_tensor<typename Op::value_type>(in.Shape(), space, stream);
      }
      else {
        bool supported = fn();

        if(supported) {
          return make_tensor<typename Op::value_type>(in.Data(), in.Descriptor());
        } else {
          return make_tensor<typename Op::value_type>(in.Shape(), space, stream);
        }
      }
    }
  }

  namespace detail {
    #ifdef MATX_EN_JIT
    // Helper function to convert to JIT storage
    // If T has ToJITStorage(), call it; otherwise return the value as-is (for scalars)
    template <typename T>
    __MATX_INLINE__ __MATX_HOST__ decltype(auto) to_jit_storage(const T& val) {
      if constexpr (has_to_jit_storage_v<T>) {
        return val.ToJITStorage();
      } else {
        return val;
      }
    }
    #endif  
  }
}; 
#endif

// RTC and nvcc
namespace matx {
  namespace detail {
    template <typename CapType, typename T, typename Func>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto ApplyGeneratorVecFunc(const Func &func, index_t index) {
      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return func(index);
      } else {

        Vector<T, static_cast<index_t>(CapType::ept)> result;
        MATX_LOOP_UNROLL
        for (int i = 0; i < static_cast<index_t>(CapType::ept); i++) {
          result.data[i] = func(index * static_cast<index_t>(CapType::ept) + i);
        }
        return result;
      }
    }

    template <typename CapType, typename OutType, typename Func, typename... Vals>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto ApplyVecFunc(const Func &func, const Vals &...vals) {
      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return func(vals...);
      } else {
        Vector<OutType, static_cast<index_t>(CapType::ept)> result;
        MATX_LOOP_UNROLL
        for (int i = 0; i < static_cast<index_t>(CapType::ept); i++) {
          result.data[i] = func(vals.data[i]...);
        }
        return result;
      }
    }

    template <typename CapType, typename ValueType>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto GetJitSentinelValue() {
      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return ValueType{};
      }
      else {
        return Vector<ValueType, static_cast<size_t>(CapType::ept)>{};
      }
    }   
  

    /**
      * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs
      *
      * @param op Operator
      * @param abs Absolute index
      * @return cuda::std::array of indices
      */
    template <typename Op>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto GetIdxFromAbs(const Op &op, index_t abs) {
      using l_stride_type = index_t;
      using l_shape_type = index_t;
      constexpr int RANK = Op::Rank();

      cuda::std::array<l_shape_type, RANK> indices;

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
      * @return cuda::std::array of indices
      */
    template <typename Op>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto BlockToIdx(const Op &op, index_t abs, int nb_dims) {
      using l_stride_type = index_t;
      using l_shape_type = index_t;
      constexpr int RANK = Op::Rank();
      cuda::std::array<l_shape_type, RANK> indices{0};

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

    template <typename T0, typename T1, typename... Tn>
    constexpr auto  __MATX_HOST__ __MATX_DEVICE__ matx_max(T0 &&t0, T1 &&t1, Tn &&... tn)
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
    }

    template <class T, class M = T>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto get_size([[maybe_unused]] T &a,
                                                [[maybe_unused]] int32_t dim)
    {
      if constexpr (is_matx_op<M>())
        return a.Size(dim);
      else
        return 1;
    }

    template <int RANK, class T, class M = T>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
    get_expanded_size([[maybe_unused]] T &a, [[maybe_unused]] int32_t dim)
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
      * @tparam CapType Capability type
      * @tparam T type of operator
      * @tparam Is type of indices
      * @param i operator
      * @param indices indices
      * @return Value after broadcasting
      */
    template <typename CapType, typename T, typename... Is>
      requires (cuda::std::conjunction_v<cuda::std::is_integral<Is>...>)
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_matx_value(T &&i, Is... indices)
    {
      constexpr int RANK = remove_cvref_t<T>::Rank();
      if constexpr (RANK == int(sizeof...(Is)) || RANK == matxNoRank) {
        // If we're only indexing with the same number of arguments as the rank of the operator, just return operator()
        return cuda::std::forward<T>(i).template operator()<CapType>(indices...);
      }
      else
      {
        // Otherwise we need to broadcast by constructing a large set of indices
        // Construct an integer sequence of the length of the tuple, but only using the last indices. We construct an offset sequence
        // to index into the broadcasted dimensions. For example, if T is a 3D tensor and we want to index as a 5D, we take the indices
        // {0, 1, 2} we'd normally index with, and add the difference in rank (2), to get {2, 3, 4}. Another way to think of this is it
        // simply chops off the first sizeof...(Is) - RANK indices since they're not used for operator().
        using seq = offset_sequence_t<sizeof...(Is) - RANK, cuda::std::make_index_sequence<RANK>>;
        auto tup = cuda::std::make_tuple(indices...);
        auto sliced_tup = select_tuple(std::forward<decltype(tup)>(tup), seq{});
        return cuda::std::apply([&] (auto... args) {                 
          return cuda::std::forward<T>(i).template operator()<CapType>(args...);
        }, sliced_tup);
      }
    }

    template <typename CapType, typename T, typename IdxType, size_t N>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_matx_value(T &&i, const cuda::std::array<IdxType, N> idx)
    {
      constexpr int RANK = remove_cvref_t<T>::Rank();
      if constexpr (RANK == N || RANK == matxNoRank) {
        // If we're only indexing with the same number of arguments as the rank of the operator, just return operator()
        return cuda::std::apply([&i](auto... args) -> decltype(auto) {
          return cuda::std::forward<T>(i).template operator()<CapType>(args...);
        }, idx);        
      }
      else
      {
        cuda::std::array<index_t, RANK> nbc_idx; // non-broadcast indices
        cuda::std::copy(idx.begin() + (N - RANK), idx.end(), nbc_idx.begin());
        return cuda::std::apply([&i](auto... args) -> decltype(auto) {
          return cuda::std::forward<T>(i).template operator()<CapType>(args...);
        }, nbc_idx);
      }
    }    


    template <typename CapType, typename T, typename... Is>
      requires (cuda::std::conjunction_v<cuda::std::is_integral<Is>...>)
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_value(T &&i, Is... indices)
    {
      if constexpr (is_matx_op<T>())
      {
        if constexpr (remove_cvref_t<T>::Rank() == 0) {
          return i.template operator()<CapType>();
        }
        else {
          return get_matx_value<CapType>(cuda::std::forward<T>(i), indices...);
        }
      }
      else
      {
        return i;
      }
    }


    template <typename CapType, typename T, typename IdxType, size_t N>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_value(T &&i, const cuda::std::array<IdxType, N> idx)
    {
      if constexpr (is_matx_op<T>())
      {
        if constexpr (remove_cvref_t<T>::Rank() == 0) {
          return i.template operator()<CapType>();
        }
        else {
          return get_matx_value<CapType, T, IdxType, N>(cuda::std::forward<T>(i), idx);
        }
      }
      else
      {
        return i;
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
  }
}

