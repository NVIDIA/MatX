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

#include <functional>
#include <cstdio>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif
#include <numeric>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/iterator.h"
#include "matx/core/operator_utils.h" 


namespace matx {

/**
 * @brief Direction for sorting
 *
 */
typedef enum { SORT_DIR_ASC, SORT_DIR_DESC } SortDirection_t;

// define of dimension size for when the cub segemented sort
// is outperformed by the radixSort
constexpr index_t cubSegmentCuttoff = 8192;
/**
 * Parameters needed to execute a sort operation.
 */
namespace detail {
typedef enum {
  CUB_OP_RADIX_SORT,
  CUB_OP_INC_SUM,
  CUB_OP_HIST_EVEN,
  CUB_OP_REDUCE,
  CUB_OP_REDUCE_SUM,
  CUB_OP_REDUCE_MIN,
  CUB_OP_REDUCE_MAX,
  CUB_OP_SELECT,
  CUB_OP_SELECT_IDX,
  CUB_OP_UNIQUE
} CUBOperation_t;

struct CubParams_t {
  CUBOperation_t op;
  std::vector<index_t> size{10};
  index_t batches;
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename T> struct HistEvenParams_t {
  T lower_level;
  T upper_level;
};

struct SortParams_t {
  SortDirection_t dir;
};

template <typename Op, typename I>
struct ReduceParams_t {
  Op reduce_op;
  I init;
};

template <typename SelectOp, typename CountTensor>
struct SelectParams_t {
  SelectOp op;
  CountTensor num_found;
};

template <typename CountTensor>
struct UniqueParams_t {
  CountTensor num_found;
};

struct EmptyParams_t {};



template <typename OutputTensor, typename InputOperator, CUBOperation_t op, typename CParams = EmptyParams_t>
class matxCubPlan_t {
  static constexpr int RANK = OutputTensor::Rank();
  using T1 = typename InputOperator::scalar_type;
  using T2 = typename OutputTensor::scalar_type;

public:
  /**
   * Construct a handle for CUB operations
   *
   * Creates a handle for performing a CUB operation. Currently supported
   * operations are sorting and prefix sum (cumsum). Operations can either be a
   * single dimension, or batched across a dimension (rows of a matrix, for
   * example).
   *
   *
   * @param a
   *   Input tensor view
   * @param a_out
   *   Sorted output
   * @param stream
   *   CUDA stream
   *
   */
  matxCubPlan_t(OutputTensor &a_out, const InputOperator &a, const CParams &cparams, const cudaStream_t stream = 0) :
    cparams_(cparams)
  {
#ifdef __CUDACC__
    // Input/output tensors much match rank/dims
    if constexpr (op == CUB_OP_RADIX_SORT || op == CUB_OP_INC_SUM) {
      static_assert(OutputTensor::Rank() == InputOperator::Rank(), "CUB input and output tensor ranks must match");
      static_assert(RANK >= 1, "CUB function must have an output rank of 1 or higher");
      for (int i = 0; i < a.Rank(); i++) {
        MATX_ASSERT(a.Size(i) == a_out.Size(i), matxInvalidSize);
      }
    }
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if constexpr (op == CUB_OP_RADIX_SORT) {
      ExecSort(a_out, a, cparams_.dir, stream);
    }
    else if constexpr (op == CUB_OP_INC_SUM) {
      ExecPrefixScanEx(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_HIST_EVEN) {
      ExecHistEven(a_out, a, cparams_.lower_level, cparams_.upper_level, stream);
    }
    else if constexpr (op == CUB_OP_REDUCE) { // General reduce
      ExecReduce(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_REDUCE_SUM) {
      ExecSum(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_REDUCE_MIN) {
      ExecMin(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_REDUCE_MAX) {
      ExecMax(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_SELECT) {
      ExecSelect(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_SELECT_IDX) {
      ExecSelectIndex(a_out, a, stream);
    }
    else if constexpr (op == CUB_OP_UNIQUE) {
      ExecUnique(a_out, a, stream);
    }
    else {
      MATX_THROW(matxNotSupported, "Invalid CUB operation");
    }

    // Allocate any workspace needed by Sort
    matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
              stream);
#endif
  }

  static auto GetCubParams([[maybe_unused]] OutputTensor &a_out,
                                  const InputOperator &a,
                                  cudaStream_t stream)
  {
    CubParams_t params;

    for (int r = 0; r < InputOperator::Rank(); r++) {
      params.size.push_back(a.Size(r));
    }

    params.op = op;
    if constexpr (op == CUB_OP_RADIX_SORT) {
      params.batches = (RANK == 1) ? 1 : a.Size(RANK - 2);
    }
    else if constexpr (op == CUB_OP_INC_SUM || op == CUB_OP_HIST_EVEN) {
      params.batches = TotalSize(a) / a.Size(a.Rank() - 1);
    } else if constexpr ( op == CUB_OP_REDUCE || 
                          op == CUB_OP_REDUCE_SUM || 
                          op == CUB_OP_REDUCE_MIN || 
                          op == CUB_OP_REDUCE_MAX) {
      
    }
    else {
      params.batches = 1;
    }

    params.dtype = TypeToInt<T1>();

    params.stream = stream;

    return params;
  }

  /**
   * Sort destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxCubPlan_t()
  {
    matxFree(d_temp);
  }

  template <typename Func>
  void RunBatches(OutputTensor &a_out, const InputOperator &a, const Func &f, int batch_offset)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    using shape_type = index_t;
    size_t total_iter = 1;
    for (int i = 0; i < InputOperator::Rank() - batch_offset; i++) {
      total_iter *= a.Size(i);
    }

    // Get total indices per batch
    index_t total_per_batch = 1;
    index_t offset = 0;
    for (int i = InputOperator::Rank() - batch_offset; i < InputOperator::Rank(); i++) {
      total_per_batch *= a.Size(i);
    }

    std::array<shape_type, InputOperator::Rank()> idx{0};

    if constexpr (is_tensor_view_v<InputOperator>) {
      if (a.IsContiguous()) {
        for (size_t iter = 0; iter < total_iter; iter++) {
          auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
      else {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        for (size_t iter = 0; iter < total_iter; iter++) {
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(RandomOperatorIterator{base, offset}, aop);
          offset += total_per_batch;

          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
    }
    else {
      for (size_t iter = 0; iter < total_iter; iter++) {
        auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

        f(RandomOperatorIterator{a, offset}, aop);
        offset += total_per_batch;

        UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
      }
    }
  }

  /**
   * Execute an inclusive prefix sum on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam T1
   *   Type of tensor
   * @param a_out
   *   Output tensor (must be an integer type)
   * @param a
   *   Input tensor
   * @param lower
   *   Lower bound on histogram
   * @param upper
   *   Upper bound on histogram
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecHistEven(OutputTensor &a_out,
                           const InputOperator &a, const T1 lower,
                           const T1 upper, const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
    if (RANK == 1 || d_temp == nullptr) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        if (a.IsContiguous()) {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a_out.Size(a_out.Rank() - 1) + 1), lower, upper,
              static_cast<int>(a.Size(a.Rank() - 1)), stream);
        }
        else {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, RandomOperatorIterator{base}, a_out.Data(),
              static_cast<int>(a_out.Size(a_out.Rank() - 1) + 1), lower, upper,
              static_cast<int>(a.Size(a.Rank() - 1)), stream);
        }
      }
      else {
        cub::DeviceHistogram::HistogramEven(
            d_temp, temp_storage_bytes, RandomOperatorIterator{a}, a_out.Data(),
            static_cast<int>(a_out.Size(a_out.Rank() - 1)  + 1), lower, upper,
            static_cast<int>(a.Size(a.Rank() - 1)), stream);
      }
    }
    else { // Batch higher dims
      auto ft = [&](auto ...p){ return cub::DeviceHistogram::HistogramEven(p...); };
      auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a_out.Size(a_out.Rank() - 1) + 1), lower, upper,
            static_cast<int>(a.Size(a.Rank() - 1)), stream);
      RunBatches(a_out, a, f, 2);
    }
#endif
  }

  /**
   * Execute an inclusive prefix sum on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam T1
   *   Type of tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecPrefixScanEx(OutputTensor &a_out,
                               const InputOperator &a,
                               const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if (RANK == 1 || d_temp == nullptr) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, a.Data(),
                                        a_out.Data(), static_cast<int>(a.Size(a.Rank()-1)),
                                        stream);
        }
        else {
          cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, RandomOperatorIterator{base},
                                        a_out.Data(), static_cast<int>(a.Size(a.Rank()-1)),
                                        stream);
        }
      }
      else {
        cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, RandomOperatorIterator{a},
                                      a_out.Data(), static_cast<int>(a.Size(a.Rank()-1)),
                                      stream);
      }
    }
    else {
        auto ft = [&](auto ...p){ cub::DeviceScan::InclusiveSum(p...); };
        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(a.Rank()-1)), stream);
        RunBatches(a_out, a, f, 1);
    }
#endif
  }

#if (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION > 14) || (CUB_MAJOR_VERSION > 1)
  /**
   * Execute an optimized sort based on newer CUB
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam T1
   *   Type of tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   * @param dir
   *   Sort order (SORT_DIR_ASC or SORT_DIR_DESC)
   *
   */
 inline void OptimizedExecSort(
                              OutputTensor &a_out,
                              const InputOperator &a,
                              const SortDirection_t dir,                   
                              const cudaStream_t stream
                              )
{

#ifdef __CUDACC__
  if constexpr (is_tensor_view_v<InputOperator>)
  {

    //////////////////////////////////////////////////////
    //////           Rank 1 Tensors               ////////
    //////////////////////////////////////////////////////
    if (RANK == 1)
    {
      if (dir == SORT_DIR_ASC)
      {
        cub::DeviceSegmentedSort::SortKeys(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)),
            1,
            BeginOffset{a}, EndOffset{a},
            stream);
      }
      else
      {

        cub::DeviceSegmentedSort::SortKeysDescending(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)),
            1,
            BeginOffset{a}, EndOffset{a},
            stream);
      }
    }

    //////////////////////////////////////////////////////
    //////           Rank 2 Tensors               ////////
    //////////////////////////////////////////////////////
    else if (RANK == 2 || d_temp == nullptr)
    {
      if (dir == SORT_DIR_ASC)
      {
        cub::DeviceSegmentedSort::SortKeys(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)),
            static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, stream);
      }
      else
      {
        cub::DeviceSegmentedSort::SortKeysDescending(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)),
            static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, stream);
      }
    }

    //////////////////////////////////////////////////////
    //////    Batching for Higher Rank Tensors    ////////
    //////////////////////////////////////////////////////
    else
    {
      constexpr int batch_offset = 2;
      using shape_type = index_t;
      size_t total_iter = 1;
      for (int i = 0; i < InputOperator::Rank() - batch_offset; i++)
      {
        total_iter *= a.Size(i);
      }

      std::array<shape_type, InputOperator::Rank()> idx{0};

      if (dir == SORT_DIR_ASC)
      {
        auto ft = [&](auto ...p){ cub::DeviceSegmentedSort::SortKeys(p...); };

        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, stream);

        for (size_t iter = 0; iter < total_iter; iter++)
        {
          auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }

      }
      else
      {
        auto ft = [&](auto ...p){ cub::DeviceSegmentedSort::SortKeysDescending(p...); };
        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, stream);

        for (size_t iter = 0; iter < total_iter; iter++)
        {
          auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
    }
  }
#endif // end CUDACC
}
#endif //cub > 1.14


/**
 * Execute a sort on a tensor
 *
 * @note Views being passed must be in row-major order
 *
 * @tparam T1
 *   Type of tensor
 * @param a_out
 *   Output tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 * @param dir
 *   Sort order (SORT_DIR_ASC or SORT_DIR_DESC)
 *
 */
inline void ExecSort(OutputTensor &a_out,
                     const InputOperator &a,
                     const SortDirection_t dir,
                     const cudaStream_t stream)
{

#ifdef __CUDACC__
  static_assert(is_tensor_view_v<InputOperator>, "Sorting only accepts tensors for now (no operators)");
  MATX_ASSERT_STR(a.IsContiguous(), matxInvalidType, "Tensor must be contiguous in memory for sorting");
  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
#if (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION  >  14) || (CUB_MAJOR_VERSION > 1)
  // use optimized segmented sort if:
  //    - it is available (cub > 1.4)
  //    - it is greater than Rank 1 Tensor
  //    - it is < 8192 in all dims
  if( ( RANK > 1 ) && ( LargestDimSize(a) < cubSegmentCuttoff) )
  {
    OptimizedExecSort(a_out,
                      a,
                      dir,
                      stream);
    return;
  }

#endif
  // legacy Implementation
  if constexpr (is_tensor_view_v<InputOperator>)
  {

    //////////////////////////////////////////////////////
    //////           Rank 1 Tensors               ////////
    //////////////////////////////////////////////////////
    if (RANK == 1)
    {
      if (dir == SORT_DIR_ASC)
      {
        cub::DeviceRadixSort::SortKeys(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)), 0, sizeof(T1) * 8, stream);
      }
      else
      {
        cub::DeviceRadixSort::SortKeysDescending(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)), 0, sizeof(T1) * 8, stream);
      }
    }

    //////////////////////////////////////////////////////
    //////           Rank 2 Tensors               ////////
    //////////////////////////////////////////////////////
    else if (RANK == 2 || d_temp == nullptr)
    {
      if (dir == SORT_DIR_ASC)
      {
        cub::DeviceSegmentedRadixSort::SortKeys(
          d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
          static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)),
          static_cast<int>(a.Size(RANK - 2)),
          BeginOffset{a}, EndOffset{a}, 0, sizeof(T1) * 8, stream);
      }
      else
      {
        cub::DeviceSegmentedRadixSort::SortKeysDescending(
            d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, 0, sizeof(T1) * 8, stream);
      }
    }

    //////////////////////////////////////////////////////
    //////    Batching for Higher Rank Tensors    ////////
    //////////////////////////////////////////////////////
    else
    {
      constexpr int batch_offset = 2;
      using shape_type = index_t;
      size_t total_iter = 1;
      for (int i = 0; i < InputOperator::Rank() - batch_offset; i++)
      {
        total_iter *= a.Size(i);
      }

      std::array<shape_type, InputOperator::Rank()> idx{0};

      if (dir == SORT_DIR_ASC)
      {

        auto ft = [&](auto ...p){ cub::DeviceSegmentedRadixSort::SortKeys(p...); };

        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, static_cast<int>(0), static_cast<int>(sizeof(T1) * 8), stream);

        for (size_t iter = 0; iter < total_iter; iter++)
        {
          auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
      else
      {

        auto ft = [&](auto ...p){ cub::DeviceSegmentedRadixSort::SortKeysDescending(p...); };
        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, static_cast<int>(0), static_cast<int>(sizeof(T1) * 8), stream);

        for (size_t iter = 0; iter < total_iter; iter++)
        {
          auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
    }
  }
#endif // end CUDACC
}



  /**
   * Execute a reduction on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam T1
   *   Type of tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param init
   *   Value to initialize with
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecReduce(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    typename detail::base_type_t<InputOperator> in_base = a;
    typename detail::base_type_t<OutputTensor> out_base = a_out;  

    // Check whether this is a segmented reduction or single-value output. Segmented reductions are any
    // type of reduction where there's not a single output, since any type of reduction can be generalized
    // to a segmented type
    if constexpr (OutputTensor::Rank() > 0) {
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) { 
          return cub::DeviceSegmentedReduce::Reduce(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, cparams_.reduce_op,
                                    cparams_.init, stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Reduce");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) { 
        return cub::DeviceReduce::Reduce(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), cparams_.reduce_op,
                                    cparams_.init, stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceReduce::Reduce");
    }    
    
#endif
  }


  /**
   * Execute a sum on an operator
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecSum(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    typename detail::base_type_t<InputOperator> in_base = a;
    typename detail::base_type_t<OutputTensor> out_base = a_out;  

    // Check whether this is a segmented reduction or single-value output. Segmented reductions are any
    // type of reduction where there's not a single output, since any type of reduction can be generalized
    // to a segmented type
    if constexpr (OutputTensor::Rank() > 0) {
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) { 
          return cub::DeviceSegmentedReduce::Sum(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Sum");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) { 
        return cub::DeviceReduce::Sum(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in ub::DeviceReduce::Sum");
    }
#endif
  }

  /**
   * Execute a min on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecMin(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    typename detail::base_type_t<InputOperator> in_base = a;
    typename detail::base_type_t<OutputTensor> out_base = a_out;  

    // Check whether this is a segmented reduction or single-value output. Segmented reductions are any
    // type of reduction where there's not a single output, since any type of reduction can be generalized
    // to a segmented type
    if constexpr (OutputTensor::Rank() > 0) {
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) { 
          return cub::DeviceSegmentedReduce::Min(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Min");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) { 
        return cub::DeviceReduce::Min(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DevicdReduce::Min");
    }
#endif
  }

  /**
   * Execute a max on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecMax(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    typename detail::base_type_t<InputOperator> in_base = a;
    typename detail::base_type_t<OutputTensor> out_base = a_out;   

    // Check whether this is a segmented reduction or single-value output. Segmented reductions are any
    // type of reduction where there's not a single output, since any type of reduction can be generalized
    // to a segmented type
    if constexpr (OutputTensor::Rank() > 0) {
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) { 
          return cub::DeviceSegmentedReduce::Max(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Max");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) { 
        return cub::DeviceReduce::Max(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream); 
      };
      auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceReduce::Max");
    }
#endif
  }



  /**
   * Execute a selection reduction on a tensor
   *
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecSelect(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if constexpr (is_tensor_view_v<InputOperator>) {
      const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
      if (a.IsContiguous()) {
        cub::DeviceSelect::If(d_temp,
                              temp_storage_bytes,
                              a.Data(),
                              a_out.Data(),
                              cparams_.num_found.Data(),
                              static_cast<int>(TotalSize(a)),
                              cparams_.op,
                              stream);
      }
      else {
        cub::DeviceSelect::If(d_temp,
                              temp_storage_bytes,
                              RandomOperatorIterator{base},
                              a_out.Data(),
                              cparams_.num_found.Data(),
                              static_cast<int>(TotalSize(a)),
                              cparams_.op,
                              stream);
      }
    }
    else {
      cub::DeviceSelect::If(d_temp,
                            temp_storage_bytes,
                            RandomOperatorIterator{a},
                            a_out.Data(),
                            cparams_.num_found.Data(),
                            static_cast<int>(TotalSize(a)),
                            cparams_.op,
                            stream);
    }
#endif
  }


  /**
   * @brief Helper class for wrapping counting iterator and data
   *
   * @tparam DataInputIterator Data input
   * @tparam SelectOp Selection operator
   */
  template<typename DataInputIterator, typename SelectOp>
  struct IndexToSelectOp
  {
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator()(index_t idx) const
    {
      return select_op(in_data[idx]);
    }

    DataInputIterator in_data;
    SelectOp select_op;
  };

  /**
   * Execute an index selection reduction on a tensor
   *
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecSelectIndex(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if constexpr (is_tensor_view_v<InputOperator>) {
      if (a.IsContiguous()) {
        cub::DeviceSelect::If(d_temp,
                              temp_storage_bytes,
                              cub::CountingInputIterator<index_t>{0},
                              a_out.Data(),
                              cparams_.num_found.Data(),
                              static_cast<int>(TotalSize(a)),
                              IndexToSelectOp<decltype(a.Data()), decltype(cparams_.op)>{a.Data(), cparams_.op},
                              stream);
      }
      else {
        tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        cub::DeviceSelect::If(d_temp,
                              temp_storage_bytes,
                              cub::CountingInputIterator<index_t>{0},
                              a_out.Data(),
                              cparams_.num_found.Data(),
                              static_cast<int>(TotalSize(a)),
                              IndexToSelectOp<decltype(RandomOperatorIterator{base}), decltype(cparams_.op)>
                                {RandomOperatorIterator{base}, cparams_.op},
                              stream);
      }
    }
    else {
      cub::DeviceSelect::If(d_temp,
                            temp_storage_bytes,
                            cub::CountingInputIterator<index_t>{0},
                            a_out.Data(),
                            cparams_.num_found.Data(),
                            static_cast<int>(TotalSize(a)),
                            IndexToSelectOp<decltype(RandomOperatorIterator{a}), decltype(cparams_.op)>
                              {RandomOperatorIterator{a}, cparams_.op},
                            stream);
    }
#endif
  }

/**
   * Execute a unique reduction on a tensor
   *
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecUnique(OutputTensor &a_out,
                       const InputOperator &a,
                       const cudaStream_t stream)
  {
#ifdef __CUDACC__
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
      
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceSelect::Unique(d_temp,
                                temp_storage_bytes,
                                a.Data(),
                                a_out.Data(),
                                cparams_.num_found.Data(),
                                static_cast<int>(TotalSize(a)),
                                stream);
        }
        else {
          cub::DeviceSelect::Unique(d_temp,
                                temp_storage_bytes,
                                RandomOperatorIterator{base},
                                a_out.Data(),
                                cparams_.num_found.Data(),
                                static_cast<int>(TotalSize(a)),
                                stream);
        }
    }
    else {
      cub::DeviceSelect::Unique(d_temp,
                            temp_storage_bytes,
                            RandomOperatorIterator{a},
                            a_out.Data(),
                            cparams_.num_found.Data(),
                            static_cast<int>(TotalSize(a)),
                            stream);
    }
#endif
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  CubParams_t params;
  CParams cparams_; ///< Parameters specific to the operation type
  T1 *d_temp = nullptr;
  int *d_histogram = nullptr; // Used for hist()
  size_t temp_storage_bytes = 0;
};


/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common Sort parameters change
 */
struct CubParamsKeyHash {
  std::size_t operator()(const CubParams_t &k) const noexcept
  {
    uint64_t shash = 0;
    for (size_t r = 0; r < k.size.size(); r++) {
      shash += std::hash<uint64_t>()(k.size[r]);
    }

    return (std::hash<uint64_t>()(k.batches)) +
           (std::hash<uint64_t>()((uint64_t)k.stream)) +
           (std::hash<uint64_t>()((uint64_t)k.op)) +
           shash;
  }
};

/**
 * Test Sort parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct CubParamsKeyEq {
  bool operator()(const CubParams_t &l, const CubParams_t &t) const noexcept
  {
    if (l.size.size() != t.size.size()) {
      return false;
    }

    for (size_t r = 0; r < l.size.size(); r++) {
      if (l.size[r] != t.size[r]) {
        return false;
      }
    }
    

    return l.batches == t.batches && l.dtype == t.dtype &&
           l.stream == t.stream && l.op == t.op;
  }
};

// Static caches of Sort handles
static matxCache_t<CubParams_t, CubParamsKeyHash, CubParamsKeyEq> cub_cache;
}


/**
 * Reduce a tensor using CUB
 *
 * Reduces a tensor using the CUB library for either a 0D or 1D output tensor. There
 * is an existing reduce() implementation as part of matx_reduce.h, but this function
 * exists in cases where CUB is more optimal/faster.
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam InputOperator
 *   Input tensor type
 * @tparam ReduceOp
 *   Reduction type
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param init
 *   Value to initialize the reduction with
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator, typename ReduceOp>
void cub_reduce(OutputTensor &a_out, const InputOperator &a, typename InputOperator::scalar_type init,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  // Get parameters required by these tensors
  using param_type = typename detail::ReduceParams_t<ReduceOp, typename InputOperator::scalar_type>;
  auto reduce_params = param_type{ReduceOp{}, init};

#ifndef MATX_DISABLE_CUB_CACHE
  // Get cache or new Sort plan if it doesn't exist
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_REDUCE,
                            param_type>::GetCubParams(a_out, a, stream);  
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type>{
        a_out, a, reduce_params, stream};

      detail::cub_cache.Insert(params, static_cast<void *>(tmp));  
    tmp->ExecReduce(a_out, a, stream);
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type> *>(
            ret.value());
    type->ExecReduce(a_out, a, stream);
  }
#else
    auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type>{
        a_out, a, reduce_params, stream};
    tmp.ExecReduce(a_out, a, stream);
#endif
#endif
}

/**
 * Sum a tensor using CUB
 *
 * Performs a sum reduction/binary fold over + from a tensor.
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam InputOperator
 *   Input tensor type
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void cub_sum(OutputTensor &a_out, const InputOperator &a,
          const cudaStream_t stream = 0)
{

#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

#ifndef MATX_DISABLE_CUB_CACHE  
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator,
                            detail::CUB_OP_REDUCE_SUM>::GetCubParams(a_out, a, stream);


  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM>{a_out, a, {}, stream};
    tmp->ExecSum(a_out, a, stream);
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));   
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM> *>(ret.value());
    type->ExecSum(a_out, a, stream);
  }
#else      
    auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM>{a_out, a, {}, stream};
    tmp.ExecSum(a_out, a, stream);
#endif    
#endif
}

/**
 * Find min of a tensor using CUB
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam InputOperator
 *   Input tensor type
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void cub_min(OutputTensor &a_out, const InputOperator &a,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

#ifndef MATX_DISABLE_CUB_CACHE  
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_REDUCE_MIN>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN>{
        a_out, a, {}, stream};

    tmp->ExecMin(a_out, a, stream);  
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));        
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN> *>(
            ret.value());
    type->ExecMin(a_out, a, stream);
  }
#else
  auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN>{
      a_out, a, {}, stream};

  tmp.ExecMin(a_out, a, stream);     
#endif  
#endif
}

/**
 * Find max of a tensor using CUB
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam InputOperator
 *   Input tensor type
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void cub_max(OutputTensor &a_out, const InputOperator &a,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
#ifndef MATX_DISABLE_CUB_CACHE
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_REDUCE_MAX>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX>{
        a_out, a, {}, stream};   
    tmp->ExecMax(a_out, a, stream); 
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));   
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX> *>(
            ret.value());
    type->ExecMax(a_out, a, stream);
  }
#else
    auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX>{
        a_out, a, {}, stream};   
    tmp.ExecMax(a_out, a, stream);     
#endif    
#endif
}

/**
 * Sort rows of a tensor
 *
 * Sort rows of a tensor using a radix sort. Currently supported types are
 * float, double, ints, and long ints (both signed and unsigned). For a 1D
 * tensor, a linear sort is performed. For 2D and above each row of the inner
 * dimensions are batched and sorted separately. There is currently a
 * restriction that the tensor must have contiguous data in both rows and
 * columns, but this restriction may be removed in the future.
 *
 * @note Temporary memory is used during the sorting process, and about 2N will
 * be allocated, where N is the length of the tensor.
 *
 * @tparam T1
 *   Type of data to sort
 * @tparam RANK
 *   Rank of tensor
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param dir
 *   Direction to sort (either SORT_DIR_ASC or SORT_DIR_DESC)
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void sort(OutputTensor &a_out, const InputOperator &a,
          const SortDirection_t dir,
          cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  
  detail::SortParams_t p{dir};

#ifndef MATX_DISABLE_CUB_CACHE  
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)>{
        a_out, a, p, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSort(a_out, a, dir, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)> *>(
            ret.value());
    sort_type->ExecSort(a_out, a, dir, stream);
  }
#else
    auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)>{
        a_out, a, p, stream};  
    tmp.ExecSort(a_out, a, dir, stream);      
#endif    
#endif
}

template <typename OutputTensor, typename InputOperator>
void sort(OutputTensor &a_out, const InputOperator &a,
          const SortDirection_t dir,
          [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  typename detail::base_type_t<InputOperator> in_base = a;
  typename detail::base_type_t<OutputTensor>  out_base = a_out;
  auto lin  = matx::RandomOperatorIterator{in_base};
  auto lout = matx::RandomOperatorOutputIterator{out_base};
  
  if constexpr (InputOperator::Rank() == 1) {
    if (dir == SORT_DIR_ASC) {
      std::partial_sort_copy( lin, 
                              lin  + a.Size(0), 
                              lout, 
                              lout + a_out.Size(0));
    }
    else {
      std::partial_sort_copy( lin, 
                              lin  + a.Size(0), 
                              lout, 
                              lout + a_out.Size(0), 
                              std::greater<typename InputOperator::scalar_type>());
    }      
  }
  else {
    for (index_t b = 0; b < lout.Size(0); b++) {
      if (dir == SORT_DIR_ASC) {
        std::partial_sort_copy( lin  + b*a.Size(1), 
                                lin  + (b+1)*a.Size(1), 
                                lout + b*a.Size(1), 
                                lout + (b+1)*a.Size(1));
      }
      else {
        std::partial_sort_copy( lin  + b*a.Size(1), 
                                lin  + (b+1)*a.Size(1), 
                                lout + b*a.Size(1), 
                                lout + (b+1)*a.Size(1),
                                std::greater<typename InputOperator::scalar_type>());
      }             
    }
  }
}

/**
 * Compute a cumulative sum (prefix sum) of rows of a tensor
 *
 * Computes an exclusive cumulative sum over rows in a tensor. For example, and
 * input tensor of [1, 2, 3, 4] would give the output [1, 3, 6, 10].
 *
 * @tparam T1
 *   Type of data to sort
 * @tparam RANK
 *   Rank of tensor
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void cumsum(OutputTensor &a_out, const InputOperator &a,
            cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();

#ifndef MATX_DISABLE_CUB_CACHE    
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp =
        new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>{a_out, a, {}, stream};   
    tmp->ExecPrefixScanEx(a_out, a, stream); 
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM> *>(ret.value());
    sort_type->ExecPrefixScanEx(a_out, a, stream);
  }
#else 
    auto tmp =
        detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>{a_out, a, {}, stream};   
    tmp.ExecPrefixScanEx(a_out, a, stream);     
#endif      
#endif
}

template <typename OutputTensor, typename InputOperator>
void cumsum(OutputTensor &a_out, const InputOperator &a,
            [[maybe_unused]] SingleThreadHostExecutor exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  typename detail::base_type_t<InputOperator> in_base = a;
  typename detail::base_type_t<OutputTensor>  out_base = a_out;
  auto lin  = matx::RandomOperatorIterator{in_base};
  auto lout = matx::RandomOperatorOutputIterator{out_base};

  if constexpr (OutputTensor::Rank() == 1) {
    std::partial_sum( lin, 
                      lin  + a.Size(0), 
                      lout);     
  }
  else if constexpr (InputOperator::Rank() == 2) {
    for (index_t b = 0; b < a.Size(0); b++) {
      std::partial_sum( lin  + b     * a.Size(1), 
                        lin  + (b+1) * a.Size(1), 
                        lout + b     * a.Size(1));             
    }
  }
  else {
    MATX_ASSERT_STR(false, matxInvalidParameter, "Only 1 and 2D cumulative sums supported on host");
  }
  
 
#endif
}

/**
 * Compute a histogram of rows in a tensor
 *
 * Computes a histogram with the given number of levels and upper/lower limits.
 * The number of levels is one greater than the number of bins generated, and is
 * determined by the size of the last dimension of the output tensor. Each bin
 * contains elements falling within idx*(upper-lower)/a.out.Lsize(). In other
 * words, each bin is as large as the different between the upper and lower
 * bounds and the number of bins
 *
 * @tparam T1
 *   Type of data to sort
 * @tparam RANK
 *   Rank of tensor
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param lower
 *   Lower limit
 * @param upper
 *   Upper limit
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void hist(OutputTensor &a_out, const InputOperator &a,
          const typename InputOperator::scalar_type lower,
          const typename InputOperator::scalar_type upper, const cudaStream_t stream = 0)
{
  static_assert(std::is_same_v<typename OutputTensor::scalar_type, int>, "Output histogram tensor must use int type");
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  detail::HistEvenParams_t<typename InputOperator::scalar_type> hp{lower, upper};
#ifndef MATX_DISABLE_CUB_CACHE   
  // Get parameters required by these tensors  
  auto params =
       detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_HIST_EVEN>::GetCubParams(a_out, a, stream);


  // Don't cache until we have a good plan for hashing parameters here
  // Get cache or new Sort plan if it doesn't exist
   auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_HIST_EVEN,
                                          detail::HistEvenParams_t<typename InputOperator::scalar_type>>{
        a_out, a, detail::HistEvenParams_t<typename InputOperator::scalar_type>{hp}, stream};
 
    tmp->ExecHistEven(a_out, a, lower, upper, stream);  
    detail::cub_cache.Insert(params, static_cast<void *>(tmp)); 
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator,
            detail::CUB_OP_HIST_EVEN, detail::HistEvenParams_t<typename InputOperator::scalar_type>> *>(
            ret.value());
    sort_type->ExecHistEven(a_out, a, lower, upper, stream);
  }
#else 
    auto tmp = detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_HIST_EVEN,
                                          detail::HistEvenParams_t<typename InputOperator::scalar_type>>{
        a_out, a, detail::HistEvenParams_t<typename InputOperator::scalar_type>{hp}, stream};
 
    tmp.ExecHistEven(a_out, a, lower, upper, stream);      
#endif    

#endif
}


// Utility functions for find()
template <typename T>
struct LT
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ LT(T c) : c_(c) {}
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a < c_);
    }
};

template <typename T>
struct GT
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ GT(T c) : c_(c) { }
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a > c_);
    }
};

template <typename T>
struct EQ
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ EQ(T c) : c_(c) {}
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a == c_);
    }
};

template <typename T>
struct NEQ
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ NEQ(T c) : c_(c) {}
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a != c_);
    }
};

template <typename T>
struct LTE
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ LTE(T c) : c_(c) {}
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a <= c_);
    }
};

template <typename T>
struct GTE
{
    T c_;
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ GTE(T c) : c_(c) {}
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
    bool operator()(const T &a) const {
        return (a >= c_);
    }
};


/**
 * Reduce values that meet a certain criteria
 *
 * Finds all values meeting the criteria specified in SelectOp, and saves them out to an output tensor. This
 * function is different from the MatX IF operator in that this performs a reduction on the input, whereas IF
 * is only for element-wise output. Output tensor must be large enough to hold unique entries. To be safe,
 * this can be the same size as the input, but if something is known about the data to indicate not as many
 * entries are needed, the output can be smaller.
 *
 * @tparam SelectType
 *   Type of select functor
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param sel
 *   Select functor
 * @param exec
 *   CUDA executor or stream
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};  
  cudaStream_t stream = exec.getStream();

#ifndef MATX_DISABLE_CUB_CACHE  

  // Get parameters required by these tensors
  auto params =
       detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT, SelectType>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);


  // Don't cache until we have a good plan for hashing parameters here
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_SELECT,
                                          decltype(cparams)>{a_out, a, cparams, stream}; 
    tmp->ExecSelect(a_out, a, stream); 
    detail::cub_cache.Insert(params, static_cast<void *>(tmp)); 
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecSelect(a_out, a, stream);
  }
#else 
    auto tmp = detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_SELECT,
                                          decltype(cparams)>{a_out, a, cparams, stream}; 
    tmp.ExecSelect(a_out, a, stream);     
#endif    
#endif
}

/**
 * Reduce values that meet a certain criteria
 *
 * Finds all values meeting the criteria specified in SelectOp, and saves them out to an output tensor. This
 * function is different from the MatX IF operator in that this performs a reduction on the input, whereas IF
 * is only for element-wise output. Output tensor must be large enough to hold unique entries. To be safe,
 * this can be the same size as the input, but if something is known about the data to indicate not as many
 * entries are needed, the output can be smaller.
 *
 * @tparam SelectType
 *   Type of select functor
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param sel
 *   Select functor
 * @param exec
 *   Single-threaded host executor
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if (a.Size(a.Rank() - 1) == 0) {
    num_found() = 0;
    return;
  }

  int cnt = 0;
  auto it = std::find_if(cbegin(a), cend(a), sel);
  while (it != cend(a)) {
    a_out(cnt++) = *it;
    it = std::find_if(++it, cend(a), sel);
  }

  num_found() = cnt;
}


/**
 * Reduce indices that meet a certain criteria
 *
 * Finds all indices of values meeting the criteria specified in SelectOp, and saves them out to an output tensor. This
 * function is different from the MatX IF operator in that this performs a reduction on the input, whereas IF
 * is only for element-wise output. Output tensor must be large enough to hold unique entries. To be safe,
 * this can be the same size as the input, but if something is known about the data to indicate not as many
 * entries are needed, the output can be smaller. 
 *
 * @tparam SelectType
 *   Type of select functor
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param sel
 *   Select functor
 * @param exec
 *   CUDA executor stream
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find_idx(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};

#ifndef MATX_DISABLE_CUB_CACHE 
  // Get parameters required by these tensors
  auto params =
       detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_IDX, SelectType>::GetCubParams(a_out, a, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);

  // Don't cache until we have a good plan for hashing parameters here
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_SELECT_IDX,
                                          decltype(cparams)>{a_out, a, cparams, stream}; 
    tmp->ExecSelectIndex(a_out, a, stream);
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_IDX, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecSelectIndex(a_out, a, stream);
  }
#else 
    auto tmp = detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_SELECT_IDX,
                                          decltype(cparams)>{a_out, a, cparams, stream}; 
    tmp.ExecSelectIndex(a_out, a, stream); 
#endif  
#endif
}

/**
 * Reduce indices that meet a certain criteria
 *
 * Finds all indices of values meeting the criteria specified in SelectOp, and saves them out to an output tensor. This
 * function is different from the MatX IF operator in that this performs a reduction on the input, whereas IF
 * is only for element-wise output. Output tensor must be large enough to hold unique entries. To be safe,
 * this can be the same size as the input, but if something is known about the data to indicate not as many
 * entries are needed, the output can be smaller. 
 *
 * @tparam SelectType
 *   Type of select functor
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param sel
 *   Select functor
 * @param exec
 *   Single host executor
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find_idx(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if (a.Size(a.Rank() - 1) == 0) {
    num_found() = 0;
    return;
  }

  int cnt = 0;
  auto it = std::find_if(cbegin(a), cend(a), sel);
  while (it != cend(a)) {
    a_out(cnt++) = static_cast<int>(it - cbegin(a));
    it = std::find_if(++it, cend(a), sel);
  }

  num_found() = cnt;
}


/**
 * Reduce to unique values
 *
 * Reduces the input to only unique values saved into the output. Output tensor must be large enough to
 * hold unique entries. To be safe, this can be the same size as the input, but if something is known about
 * the data to indicate not as many entries are needed, the output can be smaller.
 *
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 */
template <typename CountTensor, typename OutputTensor, typename InputOperator>
void unique(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a,  cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  
  // Allocate space for sorted input since CUB doesn't do unique over unsorted inputs
  typename InputOperator::scalar_type *sort_ptr;
  matxAlloc((void **)&sort_ptr, a.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto sort_tensor = make_tensor<typename InputOperator::scalar_type>(sort_ptr, a.Shape());

  matx::sort(sort_tensor, a, SORT_DIR_ASC, stream);

  auto cparams = detail::UniqueParams_t<CountTensor>{num_found};

#ifndef MATX_DISABLE_CUB_CACHE 
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_UNIQUE, decltype(cparams)>::GetCubParams(a_out, sort_tensor, stream);

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);


  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_UNIQUE,
                                          decltype(cparams)>{a_out, sort_tensor, cparams, stream};
    tmp->ExecUnique(a_out, sort_tensor, stream);
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_UNIQUE, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecUnique(a_out, sort_tensor, stream);
  }
#else 
    auto tmp = detail::matxCubPlan_t< OutputTensor,
                                          InputOperator,
                                          detail::CUB_OP_UNIQUE,
                                          decltype(cparams)>{a_out, sort_tensor, cparams, stream};
    tmp.ExecUnique(a_out, sort_tensor, stream);
#endif   
#endif
}

/**
 * Reduce to unique values
 *
 * Reduces the input to only unique values saved into the output. Output tensor must be large enough to
 * hold unique entries. To be safe, this can be the same size as the input, but if something is known about
 * the data to indicate not as many entries are needed, the output can be smaller.
 *
  * @tparam CountTensor
 *   Output items type
 * @tparam OutputTensor
 *   Output type
 * @tparam InputOperator
 *   Input type
 * @param num_found
 *   Number of items found meeting criteria
 * @param a_out
 *   Sorted tensor
 * @param a
 *   Input tensor
 * @param exec
 *   Single thread executor
 */
template <typename CountTensor, typename OutputTensor, typename InputOperator>
void unique(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, [[maybe_unused]] SingleThreadHostExecutor exec)
{
#ifdef __CUDACC__
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  if (a.Size(a.Rank() - 1) == 0) {
    num_found() = 0;
    return;
  }

  std::partial_sort_copy(cbegin(a), cend(a), begin(a_out), end(a_out));
  auto last = std::unique(begin(a_out), end(a_out));
  num_found() = static_cast<int>(last - begin(a_out));  
#endif
}
}; // namespace matx
