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
#include "matx/core/dlpack.h"
#include "matx/core/make_tensor.h"
#include "matx/kernels/utility.cuh"
#include "matx/transforms/copy.h"

namespace matx
{
  static constexpr bool PRINT_ON_DEVICE = false;      ///< print() uses printf on device
  inline unsigned int PRINT_PRECISION = 4;            ///< control PrintVal()'s precision

  /**
   * Print formatting type specifier.  Default: MATX_PRINT_FORMAT_DEFAULT
   */
  enum PrintFormatType
  {
    /// Original MATX print formatting
    MATX_PRINT_FORMAT_DEFAULT,

    /// Print formatting allowing cut&paste as MATLAB or Octave multi-dimensional matrix
    MATX_PRINT_FORMAT_MLAB,

    /// Print formatting allowing cut&paste as Python list or list of lists
    MATX_PRINT_FORMAT_PYTHON
  };
  inline enum PrintFormatType PRINT_FORMAT_TYPE = MATX_PRINT_FORMAT_DEFAULT;

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
      maxSize = cuda::std::max(op.Size(i), maxSize);
    }

    return maxSize;
  }

  namespace detail {

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
      constexpr int RANK = op.Rank();
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
     * @tparam T type of operator
     * @tparam Is type of indices
     * @param i operator
     * @param indices indices
     * @return Value after broadcasting
     */
    template <class T, typename... Is>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(const T &i, Is... indices)
    {
      if constexpr (T::Rank() == int(sizeof...(Is)) || T::Rank() == matxNoRank) {
        return i(indices...);
      }
      else
      {
        // Construct an integer sequence of the length of the tuple, but only using the last indices
        using seq = offset_sequence_t<sizeof...(Is) - T::Rank(), std::make_index_sequence<T::Rank()>>;
        auto tup = cuda::std::make_tuple(indices...);
        auto sliced_tup = select_tuple(std::forward<decltype(tup)>(tup), seq{});
        return cuda::std::apply([&](auto... args) {
          return i(args...);
        }, sliced_tup);
      }
    }


    template <class T, typename... Is>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(const T &i, Is... indices)
    {
      if constexpr (is_matx_op<T>())
      {
        return get_matx_value(i, indices...);
      }
      else
      {
        return i;
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
    void UpdateIndices(const T& op, cuda::std::array<I, R> &idx, int res) {
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


    template <typename T> constexpr DLDataType TypeToDLPackType()
    {
      if constexpr (std::is_same_v<T, cuda::std::complex<float>>)
        return {kDLComplex, 64, 1};
      if constexpr (std::is_same_v<T, cuda::std::complex<double>>)
        return {kDLComplex, 128, 1};
      if constexpr (std::is_same_v<T, matxFp16>)
        return {kDLFloat, 16, 1};
      if constexpr (std::is_same_v<T, matxBf16>)
        return {kDLBfloat, 16, 1};
      if constexpr (std::is_same_v<T, matxFp16Complex>)
        return {kDLComplex, 32, 1};
      if constexpr (std::is_same_v<T, matxBf16Complex>)
        return {kDLComplex, 32, 1}; // Wrong, but no other choice
      if constexpr (std::is_same_v<T, float>)
        return {kDLFloat, 32, 1};
      if constexpr (std::is_same_v<T, double>)
        return {kDLFloat, 64, 1};
      if constexpr (std::is_same_v<T, int8_t>)
        return {kDLInt, 8, 1};
      if constexpr (std::is_same_v<T, int16_t>)
        return {kDLInt, 16, 1};
      if constexpr (std::is_same_v<T, int32_t>)
        return {kDLInt, 32, 1};
      if constexpr (std::is_same_v<T, int64_t>)
        return {kDLInt, 64, 1};
      if constexpr (std::is_same_v<T, uint8_t>)
        return {kDLUInt, 8, 1};
      if constexpr (std::is_same_v<T, uint16_t>)
        return {kDLUInt, 16, 1};
      if constexpr (std::is_same_v<T, uint32_t>)
        return {kDLUInt, 32, 1};
      if constexpr (std::is_same_v<T, uint64_t>)
        return {kDLUInt, 64, 1};
      if constexpr (std::is_same_v<T, bool>)
  #if DLPACK_VERSION >= 80
        return {kDLBool, 8, 1};
  #else
        return {kDLUInt, 8, 1};
  #endif

      return {kDLOpaqueHandle, 1, 1};
    }


  template <typename Op, typename Executor>
  auto OpToTensor(Op &&op, [[maybe_unused]] const Executor &exec) {
    if constexpr (is_matx_transform_op<Op>()) {
      // We can assume that if a transform is passed to the input then PreRun has already completed
      // on the transform and we can use the internal pointer
      return make_tensor<typename Op::value_type>(op.Data(), Shape(op));
    }    
    else if constexpr (!is_tensor_view_v<Op>) {
      if constexpr (is_cuda_executor_v<Executor>) {
        return make_tensor<typename remove_cvref<Op>::value_type>(op.Shape(), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
      } else {
        return make_tensor<typename remove_cvref<Op>::value_type>(op.Shape(), MATX_HOST_MALLOC_MEMORY);
      }
    } else {
      return op;
    }
  }

  /**
   * Get a transposed view of a tensor or operator into a user-supplied buffer
   *
   * @param tp
   *   Pointer to pre-allocated memory
   * @param a
   *   Tensor to transpose
   * @param exec
   *   Executor
   */
  template <typename TensorType, typename Executor>
  __MATX_INLINE__ auto
  TransposeCopy(typename TensorType::value_type *tp, const TensorType &a, const Executor &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    auto pa = transpose_matrix(a);
    auto tv = make_tensor(tp, pa.Shape());
    matx::copy(tv, pa, exec);
    return tv;
  }

  }

}
