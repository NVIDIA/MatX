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
#include <numeric>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif // __CUDACC__

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/iterator.h"
#include "matx/core/operator_utils.h"
#include "matx/transforms/cccl_iterators.h"


namespace matx {

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
  CUB_OP_SELECT_VALS,
  CUB_OP_SELECT_IDX,
  CUB_OP_UNIQUE,
  CUB_OP_SINGLE_ARG_REDUCE,
  CUB_OP_DUAL_ARG_REDUCE,
} CUBOperation_t;

struct CubParams_t {
  CUBOperation_t op;
  std::vector<index_t> size{10};
  index_t batches{0};
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename T> struct HistEvenParams_t {
  T lower_level;
  T upper_level;
  int num_levels;
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
  using T1 = typename InputOperator::value_type;
  using T2 = typename OutputTensor::value_type;

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
      ExecHistEven(a_out, a, cparams_.lower_level, cparams_.upper_level, cparams_.num_levels, stream);
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
    else if constexpr (op == CUB_OP_SELECT_VALS) {
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
    matxFree(d_temp, cudaStreamDefault);
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

    cuda::std::array<shape_type, InputOperator::Rank()> idx{0};

    if constexpr (is_tensor_view_v<InputOperator>) {
      if (a.IsContiguous()) {
        for (size_t iter = 0; iter < total_iter; iter++) {
          auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = cuda::std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(ap, aop);

          // Update all but the last batch_offset indices
          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
      else {
        const tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        for (size_t iter = 0; iter < total_iter; iter++) {
          auto aop = cuda::std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

          f(RandomOperatorIterator{base, offset}, aop);
          offset += total_per_batch;

          UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
        }
      }
    }
    else {
      for (size_t iter = 0; iter < total_iter; iter++) {
        auto aop = cuda::std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

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
   * @param num_levels
   *   Number of levels in histogram
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecHistEven(OutputTensor &a_out,
                           const InputOperator &a, const T1 lower,
                           const T1 upper, int num_levels, const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
    if (RANK == 1 || d_temp == nullptr) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        if (a.IsContiguous()) {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              num_levels, lower, upper,
              static_cast<int>(a.Size(a.Rank() - 1)), stream);
        }
        else {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, RandomOperatorIterator{base}, a_out.Data(),
              num_levels, lower, upper,
              static_cast<int>(a.Size(a.Rank() - 1)), stream);
        }
      }
      else {
        cub::DeviceHistogram::HistogramEven(
            d_temp, temp_storage_bytes, RandomOperatorIterator{a}, a_out.Data(),
            num_levels, lower, upper,
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
        const tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
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
     // CUB added support for large items and segments in https://github.com/NVIDIA/cccl/pull/3308
    int64_t _num_segments = 1;
    for (int i = 0; i < RANK - 1; i++) {
      _num_segments *= a.Size(i);
    }
    int64_t _num_items = _num_segments * a.Size(RANK - 1);
#if 0  // TODO: add conditional on CUB_MAJOR_VERSION once released
    int64_t num_segments = _num_segments;
    int64_t num_items = _num_items;
#else
    if (_num_items > std::numeric_limits<int>::max()) {
      std::string err_msg = "Sorting is not supported for tensors with more than 2^" + std::to_string(std::numeric_limits<int>::digits) + " items";
      MATX_THROW(matxInvalidSize, err_msg);
    }
    int num_segments = static_cast<int>(_num_segments);
    int num_items = static_cast<int>(_num_items);
#endif
    if (dir == SORT_DIR_ASC)
    {
      cub::DeviceSegmentedSort::SortKeys(
          d_temp, temp_storage_bytes,
          a.Data(), a_out.Data(),
          num_items,
          num_segments,
          BeginOffset{a}, EndOffset{a}, stream);
    }
    else
    {
      cub::DeviceSegmentedSort::SortKeysDescending(
          d_temp, temp_storage_bytes,
          a.Data(), a_out.Data(),
          num_items,
          num_segments,
          BeginOffset{a}, EndOffset{a}, stream);
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
  if (! a.IsContiguous()) {
    MATX_THROW(matxInvalidType, "Tensor must be contiguous in memory for sorting");
  }

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

      cuda::std::array<shape_type, InputOperator::Rank()> idx{0};

      if (dir == SORT_DIR_ASC)
      {

        auto ft = [&](auto ...p){ cub::DeviceSegmentedRadixSort::SortKeys(p...); };

        auto f = std::bind(ft, d_temp, temp_storage_bytes, std::placeholders::_1, std::placeholders::_2, static_cast<int>(a.Size(RANK-1)*a.Size(RANK-2)), static_cast<int>(a.Size(RANK - 2)),
            BeginOffset{a}, EndOffset{a}, static_cast<int>(0), static_cast<int>(sizeof(T1) * 8), stream);

        for (size_t iter = 0; iter < total_iter; iter++)
        {
          auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = cuda::std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

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
          auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
          auto aop = cuda::std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

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
#if CUB_MAJOR_VERSION >= 3 && CUB_MINOR_VERSION >= 2
      [[maybe_unused]] cudaError_t err;
      if constexpr(is_tensor_view_v<InputOperator>) {
        if( a.IsContiguous() && a_out.IsContiguous()) {
          const int seg_size = static_cast<int>(TotalSize(a) / TotalSize(out_base));
          err = cub::DeviceSegmentedReduce::Reduce(d_temp, temp_storage_bytes, in_base.Data(), out_base.Data(), static_cast<cuda::std::int64_t>(TotalSize(out_base)), seg_size, cparams_.reduce_op,
                                                  cparams_.init, stream);
          MATX_ASSERT_STR_EXP(err, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Reduce");
          return;
        }
      }

      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) {
        return cub::DeviceSegmentedReduce::Reduce(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, cparams_.reduce_op,
                                  cparams_.init, stream);
      };
      err = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(err, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Reduce");
#else
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) {
          return cub::DeviceSegmentedReduce::Reduce(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, cparams_.reduce_op,
                                    cparams_.init, stream);
      };
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Reduce");
#endif
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) {
        return cub::DeviceReduce::Reduce(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), cparams_.reduce_op,
                                    cparams_.init, stream);
      };
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
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
      // Check if fixed-size reductions are supported
#if CUB_MAJOR_VERSION >= 3 && CUB_MINOR_VERSION >= 2
      [[maybe_unused]] cudaError_t err;
      if constexpr (is_tensor_view_v<InputOperator>) {
        if(a.IsContiguous() && a_out.IsContiguous()) {
          const int seg_size = static_cast<int>(TotalSize(a) / TotalSize(out_base));
          err = cub::DeviceSegmentedReduce::Sum(d_temp, temp_storage_bytes, in_base.Data(), out_base.Data(), static_cast<cuda::std::int64_t>(TotalSize(out_base)), seg_size, stream);
          MATX_ASSERT_STR_EXP(err, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Sum");
          return;
        }
      }

      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) {
          return cub::DeviceSegmentedReduce::Sum(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, stream);
      };
      err = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(err, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Sum");
#else
      auto ft = [&](auto &&in, auto &&out, auto &&begin, auto &&end) {
        return cub::DeviceSegmentedReduce::Sum(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(out_base)), begin, end, stream);
      };
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Sum");
#endif
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) {
        return cub::DeviceReduce::Sum(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream);
      };
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
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
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Min");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) {
        return cub::DeviceReduce::Min(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream);
      };
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
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
      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
      MATX_ASSERT_STR_EXP(rv, cudaSuccess, matxCudaError, "Error in cub::DeviceSegmentedReduce::Max");
    }
    else {
      auto ft = [&](auto &&in, auto &&out, [[maybe_unused]] auto &&unused1, [[maybe_unused]] auto &&unused2) {
        return cub::DeviceReduce::Max(d_temp, temp_storage_bytes, in, out, static_cast<int>(TotalSize(in_base)), stream);
      };

      [[maybe_unused]] auto rv = ReduceInput(ft, out_base, in_base);
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
      const tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
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

    if (!has_index_cmp_op_v<decltype(cparams_.op)>) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        if (a.IsContiguous()) {
          cub::DeviceSelect::If(d_temp,
                                temp_storage_bytes,
                                detail::counting_iterator<index_t>(0),
                                a_out.Data(),
                                cparams_.num_found.Data(),
                                static_cast<int>(TotalSize(a)),
                                IndexToSelectOp<decltype(a.Data()), decltype(cparams_.op)>{a.Data(), cparams_.op},
                                stream);
        }
        else {
          tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
          cub::DeviceSelect::If(d_temp,
                                temp_storage_bytes,
                                detail::counting_iterator<index_t>(0),
                                a_out.Data(),
                                cparams_.num_found.Data(),
                                static_cast<int>(TotalSize(a)),
                                IndexToSelectOp<decltype(RandomOperatorIterator{base}), decltype(cparams_.op)>
                                  {RandomOperatorIterator{base}, cparams_.op},
                                stream);
        }
      }
      else {
        tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        cub::DeviceSelect::If(d_temp,
                              temp_storage_bytes,
                              detail::counting_iterator<index_t>(0),
                              a_out.Data(),
                              cparams_.num_found.Data(),
                              static_cast<int>(TotalSize(a)),
                              IndexToSelectOp<decltype(RandomOperatorIterator{base}), decltype(cparams_.op)>
                                {RandomOperatorIterator{base}, cparams_.op},
                              stream);
      }
    }
    else {
      // Custom compare op that only takes an index. This can be more powerful for users by allowing them to define whatever
      // they want inside the op and not be limited to simple binary comparisons.
      cub::DeviceSelect::If(d_temp,
        temp_storage_bytes,
        detail::counting_iterator<index_t>(0),
        a_out.Data(),
        cparams_.num_found.Data(),
        static_cast<int>(TotalSize(a)),
        cparams_.op,
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
        const tensor_impl_t<typename InputOperator::value_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
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
  cudaStream_t stream_;
  CParams cparams_; ///< Parameters specific to the operation type
  T1 *d_temp = nullptr;
  int *d_histogram = nullptr; // Used for hist()
  size_t temp_storage_bytes = 0;
};

#ifdef __CUDACC__
struct CustomArgMaxCmp
{
  template <typename T>
  __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(const T &a, const T &b) const {
    return thrust::get<1>(a) < thrust::get<1>(b) ? b : a;
  }
};

struct CustomArgMinCmp
{
  template <typename T>
  __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(const T &a, const T &b) const {
    return thrust::get<1>(a) >= thrust::get<1>(b) ? b : a;
  }
};

struct CustomArgMinMaxCmp
{
  template <typename T>
  __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(const T &a, const T &b) const {
    T result;

    // Min part
    if (thrust::get<1>(a) >= thrust::get<1>(b))
    {
      thrust::get<0>(result) = thrust::get<0>(b);
      thrust::get<1>(result) = thrust::get<1>(b);
    }
    else
    {
      thrust::get<0>(result) = thrust::get<0>(a);
      thrust::get<1>(result) = thrust::get<1>(a);
    }

    // Max part
    if (thrust::get<3>(a) < thrust::get<3>(b))
    {
      thrust::get<2>(result) = thrust::get<2>(b);
      thrust::get<3>(result) = thrust::get<3>(b);
    }
    else
    {
      thrust::get<2>(result) = thrust::get<2>(a);
      thrust::get<3>(result) = thrust::get<3>(a);
    }

    return result;
  }
};
#endif

template <typename OutputTensor, typename TensorIndexType, typename InputOperator, typename CParams = EmptyParams_t>
class matxCubSingleArgPlan_t {
  using T1 = typename InputOperator::value_type;

public:
  matxCubSingleArgPlan_t(OutputTensor &a_out, TensorIndexType &aidx_out, const InputOperator &a, CUBOperation_t op, const CParams &cparams, const cudaStream_t stream = 0) :
    cparams_(cparams)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    if (op == CUB_OP_SINGLE_ARG_REDUCE) {
      ExecArgReduce(a_out, aidx_out, a, stream);
    }
    else {
      MATX_THROW(matxNotSupported, "Invalid CUB operation");
    }

    // Allocate any workspace needed by underly CUB algorithm
    matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
              stream);
#endif
  }

  static auto GetCubParams([[maybe_unused]] OutputTensor &a_out,
                           [[maybe_unused]] TensorIndexType &aidx_out,
                           const InputOperator &a,
                           CUBOperation_t op,
                           cudaStream_t stream)
  {
    CubParams_t params;

    for (int r = 0; r < InputOperator::Rank(); r++) {
      params.size.push_back(a.Size(r));
    }

    params.op = op;
    if constexpr (OutputTensor::Rank() > 0)
    {
      params.batches = TotalSize(a_out);
    }
    else
    {
      params.batches = 1;
    }
    params.dtype = TypeToInt<T1>();
    params.stream = stream;

    return params;
  }

  /**
   * Execute an arg reduce on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam TensorIndexType
   *   Type of the output index tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a_out
   *   Output tensor
   * @param aidx_out
   *   Output index tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecArgReduce(OutputTensor &a_out,
                            TensorIndexType &aidx_out,
                            const InputOperator &a,
                            const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const auto a_iter = matx::RandomOperatorThrustIterator{a};
    const auto zipped_input = detail::make_zip_iterator(detail::make_counting_iterator<matx::index_t>(0), a_iter);
    const auto zipped_output = detail::make_zip_iterator(aidx_out.Data(), a_out.Data());

    if constexpr (OutputTensor::Rank() > 0) {
      const int BATCHES = static_cast<int>(TotalSize(a_out));
      const int N = static_cast<int>(TotalSize(a)) / BATCHES;

      const auto r0 = matx::range<0>({BATCHES},0,N);
      const auto r0_iter = matx::RandomOperatorIterator{r0};
      const auto r1 = matx::range<0>({BATCHES},N,N);
      const auto r1_iter = matx::RandomOperatorIterator{r1};

      cub::DeviceSegmentedReduce::Reduce(
        d_temp,
        temp_storage_bytes,
        zipped_input,
        zipped_output,
        BATCHES,
        r0_iter,
        r1_iter,
        cparams_.reduce_op,
        cparams_.init,
        stream);
    }
    else {
      const int N = static_cast<int>(TotalSize(a));

      cub::DeviceReduce::Reduce(
        d_temp,
        temp_storage_bytes,
        zipped_input,
        zipped_output,
        N,
        cparams_.reduce_op,
        cparams_.init,
        stream);
    }
#endif
  }

  /**
   * Destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxCubSingleArgPlan_t()
  {
    matxFree(d_temp, cudaStreamDefault);
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  cudaStream_t stream_;
  CParams cparams_; ///< Parameters specific to the operation type
  uint8_t *d_temp = nullptr;
  size_t temp_storage_bytes = 0;
};

template <typename OutputTensor, typename TensorIndexType, typename InputOperator, typename CParams = EmptyParams_t>
class matxCubDualArgPlan_t {
  using T1 = typename InputOperator::value_type;

public:
  matxCubDualArgPlan_t(OutputTensor &a1_out,
                       TensorIndexType &aidx1_out,
                       OutputTensor &a2_out,
                       TensorIndexType &aidx2_out,
                       const InputOperator &a,
                       CUBOperation_t op,
                       const CParams &cparams,
                       const cudaStream_t stream = 0) :
    cparams_(cparams)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    if (op == CUB_OP_DUAL_ARG_REDUCE) {
      ExecDualArgReduce(a1_out, aidx1_out, a2_out, aidx2_out, a, stream);
    }
    else {
      MATX_THROW(matxNotSupported, "Invalid CUB operation");
    }

    // Allocate any workspace needed by underly CUB algorithm
    matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
              stream);
#endif
  }

  static auto GetCubParams([[maybe_unused]] OutputTensor &a1_out,
                           [[maybe_unused]] TensorIndexType &aidx1_out,
                           [[maybe_unused]] OutputTensor &a2_out,
                           [[maybe_unused]] TensorIndexType &aidx2_out,
                           const InputOperator &a,
                           CUBOperation_t op,
                           cudaStream_t stream)
  {
    CubParams_t params;

    for (int r = 0; r < InputOperator::Rank(); r++) {
      params.size.push_back(a.Size(r));
    }

    params.op = op;
    if constexpr (OutputTensor::Rank() > 0)
    {
      params.batches = TotalSize(a1_out);
    }
    else
    {
      params.batches = 1;
    }
    params.dtype = TypeToInt<T1>();
    params.stream = stream;

    return params;
  }

  /**
   * Execute an 2 value / 2 index arg reduce on a tensor
   *
   * @note Views being passed must be in row-major order
   *
   * @tparam OutputTensor
   *   Type of output tensor
   * @tparam TensorIndexType
   *   Type of the output index tensor
   * @tparam InputOperator
   *   Type of input tensor
   * @param a1_out
   *   Output first tensor
   * @param aidx1_out
   *   Output first index tensor
   * @param a2_out
   *   Output second tensor
   * @param aidx2_out
   *   Output second index tensor
   * @param a
   *   Input tensor
   * @param stream
   *   CUDA stream
   *
   */
  inline void ExecDualArgReduce(OutputTensor &a1_out,
                                TensorIndexType &aidx1_out,
                                OutputTensor &a2_out,
                                TensorIndexType &aidx2_out,
                                const InputOperator &a,
                                const cudaStream_t stream)
  {
#ifdef __CUDACC__
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const auto a_iter = matx::RandomOperatorThrustIterator{a};
    const auto zipped_input = detail::make_zip_iterator(detail::make_counting_iterator<matx::index_t>(0),
                                                        a_iter,
                                                        detail::make_counting_iterator<matx::index_t>(0),
                                                        a_iter);
    const auto zipped_output = detail::make_zip_iterator(aidx1_out.Data(), a1_out.Data(), aidx2_out.Data(), a2_out.Data());

    if constexpr (OutputTensor::Rank() > 0) {
      const int BATCHES = static_cast<int>(TotalSize(a1_out));
      const int N = static_cast<int>(TotalSize(a)) / BATCHES;

      const auto r0 = matx::range<0>({BATCHES},0,N);
      const auto r0_iter = matx::RandomOperatorIterator{r0};
      const auto r1 = matx::range<0>({BATCHES},N,N);
      const auto r1_iter = matx::RandomOperatorIterator{r1};

      cub::DeviceSegmentedReduce::Reduce(
        d_temp,
        temp_storage_bytes,
        zipped_input,
        zipped_output,
        BATCHES,
        r0_iter,
        r1_iter,
        cparams_.reduce_op,
        cparams_.init,
        stream);
    }
    else {
      const int N = static_cast<int>(TotalSize(a));

      cub::DeviceReduce::Reduce(
        d_temp,
        temp_storage_bytes,
        zipped_input,
        zipped_output,
        N,
        cparams_.reduce_op,
        cparams_.init,
        stream);
    }
#endif
  }


  /**
   * Destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxCubDualArgPlan_t()
  {
    matxFree(d_temp, cudaStreamDefault);
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  cudaStream_t stream_;
  CParams cparams_; ///< Parameters specific to the operation type
  uint8_t *d_temp = nullptr;
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

using cub_cache_t = std::unordered_map<CubParams_t, std::any, CubParamsKeyHash, CubParamsKeyEq>;

/**
 * Inner function for the public sort_impl(). sort_impl() allocates a temporary
 * tensor if needed so that the inner function can assume contiguous tensor views
 * as inputs.
 */
template <typename OutputTensor, typename InputOperator>
void sort_impl_inner(OutputTensor &a_out, const InputOperator &a,
          const SortDirection_t dir,
          cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();

  detail::SortParams_t p{dir};

#ifndef MATX_DISABLE_CUB_CACHE
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_RADIX_SORT>::GetCubParams(a_out, a, stream);

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, detail::SortParams_t>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB radix sort transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, p, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecSort(a_out, a, dir, stream);
      },
      exec
    );
#else
  auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)>{
      a_out, a, p, stream};
  tmp.ExecSort(a_out, a, dir, stream);
#endif
#endif
}


/**
 * Inner function for the public argsort_impl(). argsort_impl() allocates a temporary
 * tensor that is contiguous, and can be mutated.
 */
template <typename OutputIndexTensor, typename InputIndexTensor, typename OutputKeyTensor,  typename InputKeyTensor>
void sort_pairs_impl_inner(OutputIndexTensor &idx_out, const InputIndexTensor &idx_in,
          OutputKeyTensor &a_out, const InputKeyTensor &a_in,
          const SortDirection_t dir,
          [[maybe_unused]] cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  static constexpr int RANK = OutputIndexTensor::Rank();
  using T1 = typename InputKeyTensor::value_type;

  cudaStream_t stream = exec.getStream();
  void *d_temp = nullptr;
  size_t temp_storage_bytes = 0;

  if constexpr (RANK == 1) {
    if (dir == SORT_DIR_ASC) {
      // First call to get size
      cub::DeviceRadixSort::SortPairs(d_temp, temp_storage_bytes,
                                a_in.Data(), a_out.Data(),
                                idx_in.Data(), idx_out.Data(),
                                a_in.Size(0), 0, sizeof(T1) * 8, stream);
      matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
                stream);

      // Run sort
      cub::DeviceRadixSort::SortPairs(d_temp, temp_storage_bytes,
                                a_in.Data(), a_out.Data(),
                                idx_in.Data(), idx_out.Data(),
                                     a_in.Size(0), 0, sizeof(T1) * 8, stream);

      matxFree(d_temp, stream);
    }
    else {
      cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_storage_bytes,
                                a_in.Data(), a_out.Data(),
                                idx_in.Data(), idx_out.Data(),
                                     a_in.Size(0), 0, sizeof(T1) * 8, stream);

      // Allocate temporary storage
      matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
                stream);

      // Run sort
      cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_storage_bytes,
                                a_in.Data(), a_out.Data(),
                                idx_in.Data(), idx_out.Data(),
                                     a_in.Size(0), 0, sizeof(T1) * 8, stream);

      matxFree(d_temp, stream);
    }
  }
  else {
    // CUB added support for large items and segments in https://github.com/NVIDIA/cccl/pull/3308
    int64_t _num_segments = 1;
    for (int i = 0; i < RANK - 1; i++) {
      _num_segments *= a_in.Size(i);
    }
    int64_t _num_items = _num_segments * a_in.Size(RANK - 1);
#if 0  // TODO: add conditional on CUB_MAJOR_VERSION once released
    int64_t num_segments = _num_segments;
    int64_t num_items = _num_items;
#else
    if (_num_items > std::numeric_limits<int>::max()) {
      std::string err_msg = "Sorting is not supported for tensors with more than 2^" + std::to_string(std::numeric_limits<int>::digits) + " items";
      MATX_THROW(matxInvalidSize, err_msg);
    }
    int num_segments = static_cast<int>(_num_segments);
    int num_items = static_cast<int>(_num_items);
#endif
    if (dir == SORT_DIR_ASC)
      {
        cub::DeviceSegmentedSort::SortPairs(
            d_temp, temp_storage_bytes,
            a_in.Data(), a_out.Data(),
            idx_in.Data(), idx_out.Data(),
            num_items,
            num_segments,
            BeginOffset{a_in}, EndOffset{a_in}, stream);

        matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
                stream);

        cub::DeviceSegmentedSort::SortPairs(
            d_temp, temp_storage_bytes,
            a_in.Data(), a_out.Data(),
            idx_in.Data(), idx_out.Data(),
            num_items,
            num_segments,
            BeginOffset{a_in}, EndOffset{a_in}, stream);

        matxFree(d_temp, stream);
      }
      else
      {
        cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp, temp_storage_bytes,
            a_in.Data(), a_out.Data(),
            idx_in.Data(), idx_out.Data(),
            num_items,
            num_segments,
            BeginOffset{a_in}, EndOffset{a_in}, stream);

        matxAlloc((void **)&d_temp, temp_storage_bytes, MATX_ASYNC_DEVICE_MEMORY,
                stream);

        cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp, temp_storage_bytes,
            a_in.Data(), a_out.Data(),
            idx_in.Data(), idx_out.Data(),
            num_items,
            num_segments,
            BeginOffset{a_in}, EndOffset{a_in}, stream);

        matxFree(d_temp, stream);
      }
    }
#endif
}


template <typename Op>
__MATX_INLINE__ auto getCubArgReduceSupportedTensor( const Op &in, cudaStream_t stream) {
  // This would be better as a templated lambda, but we don't have those in C++17 yet
  const auto support_func = []() {
    return true;
  };

  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

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
void cub_reduce(OutputTensor &a_out, const InputOperator &a, typename InputOperator::value_type init,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  // Get parameters required by these tensors
  using param_type = typename detail::ReduceParams_t<ReduceOp, typename InputOperator::value_type>;
  auto reduce_params = param_type{ReduceOp{}, init};

#ifndef MATX_DISABLE_CUB_CACHE
  // Get cache or new Sort plan if it doesn't exist
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_REDUCE,
                            param_type>::GetCubParams(a_out, a, stream);
  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB reduce transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
    cache_id,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(a_out, a, reduce_params, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->ExecReduce(a_out, a, stream);
    }
  );

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
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_REDUCE_SUM>::GetCubParams(a_out, a, stream);

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM, detail::EmptyParams_t>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB reduce sum transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, detail::EmptyParams_t{}, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecSum(a_out, a, stream);
      }
    );
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

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN, detail::EmptyParams_t>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB reduce min transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, detail::EmptyParams_t{}, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecMin(a_out, a, stream);
      }
    );
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

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX, detail::EmptyParams_t>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB reduce max transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, detail::EmptyParams_t{}, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecMax(a_out, a, stream);
      }
    );
#else
  auto tmp = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX>{
      a_out, a, {}, stream};
  tmp.ExecMax(a_out, a, stream);
#endif
#endif
}

/**
 * Find index and value of custom reduction of an operator using CUB
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam TensorIndexType
 *   Output tensor index type
 * @tparam InputOperator
 *   Input operator type
 * @tparam CParams
 *   Custom reduction parameters type
 * @param a_out
 *   Output value tensor
 * @param aidx_out
 *   Output value index tensor
 * @param a
 *   Input tensor
 * @param reduce_params
 *   Reduction configuration parameters
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename TensorIndexType, typename InputOperator, typename CParams>
void cub_argreduce(OutputTensor &a_out, TensorIndexType &aidx_out, const InputOperator &a, const CParams& reduce_params,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  // converts operators to tensors (if necessary)
  auto a_out_supported = getCubArgReduceSupportedTensor(a_out, stream);
  auto aidx_out_supported = getCubArgReduceSupportedTensor(aidx_out, stream);
  auto a_supported = getCubArgReduceSupportedTensor(a, stream);

  if(!a_supported.isSameView(a)) {
    (a_supported = a).run(stream);
  }

  using cache_val_type = detail::matxCubSingleArgPlan_t<
      decltype(a_out_supported),
      decltype(aidx_out_supported),
      decltype(a_supported),
      CParams>;

  #ifndef MATX_DISABLE_CUB_CACHE
    auto params = cache_val_type::GetCubParams(a_out_supported, aidx_out_supported, a_supported, detail::CUB_OP_SINGLE_ARG_REDUCE, stream);

    auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
    MATX_LOG_DEBUG("CUB single arg reduce transform: cache_id={}", cache_id);
    detail::GetCache().LookupAndExec<detail::cub_cache_t>(
        cache_id,
        params,
        [&]() {
          return std::make_shared<cache_val_type>(a_out_supported, aidx_out_supported, a_supported, reduce_params, stream);
        },
        [&](std::shared_ptr<cache_val_type> ctype) {
          ctype->ExecArgReduce(a_out_supported, aidx_out_supported, a_supported, stream);
        }
      );
  #else
    auto tmp = cache_val_type{a_out_supported, aidx_out_supported, a_supported, detail::CUB_OP_SINGLE_ARG_REDUCE, reduce_params, stream};
    tmp.ExecArgReduce(a_out_supported, aidx_out_supported, a_supported, stream);
  #endif

  // Copy output tensors back to operators (if necessary)
  if(!a_out_supported.isSameView(a_out)) {
    (a_out = a_out_supported).run(stream);
  }

  if(!aidx_out_supported.isSameView(aidx_out)) {
    (aidx_out = aidx_out_supported).run(stream);
  }
#endif
}

/**
 * Find two indices and values of custom reduction of an operator using CUB
 *
 * @tparam OutputTensor
 *   Output tensor type
 * @tparam TensorIndexType
 *   Output tensor index type
 * @tparam InputOperator
 *   Input operator type
 * @tparam CParams
 *   Custom reduction parameters type
 * @param a1_out
 *   Output first value tensor
 * @param aidx1_out
 *   Output first value index tensor
 * @param a2_out
 *   Output second value tensor
 * @param aidx2_out
 *   Output second value index tensor
 * @param a
 *   Input tensor
 * @param reduce_params
 *   Reduction configuration parameters
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename TensorIndexType, typename InputOperator, typename CParams>
void cub_dualargreduce(OutputTensor &a1_out,
                       TensorIndexType &aidx1_out,
                       OutputTensor &a2_out,
                       TensorIndexType &aidx2_out,
                       const InputOperator &a,
                       const CParams& reduce_params,
                       const cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  using cache_val_type = detail::matxCubDualArgPlan_t<OutputTensor, TensorIndexType, InputOperator, CParams>;

  #ifndef MATX_DISABLE_CUB_CACHE
    auto params = cache_val_type::GetCubParams(a1_out, aidx1_out, a2_out, aidx2_out, a, detail::CUB_OP_DUAL_ARG_REDUCE, stream);

    auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
    MATX_LOG_DEBUG("CUB dual arg reduce transform: cache_id={}", cache_id);
    detail::GetCache().LookupAndExec<detail::cub_cache_t>(
        cache_id,
        params,
        [&]() {
          return std::make_shared<cache_val_type>(a1_out, aidx1_out, a2_out, aidx2_out, a, reduce_params, stream);
        },
        [&](std::shared_ptr<cache_val_type> ctype) {
          ctype->ExecDualArgReduce(a1_out, aidx1_out, a2_out, aidx2_out, a, stream);
        }
      );
  #else
    auto tmp = cache_val_type{a1_out, aidx1_out, a2_out, aidx2_out, a, detail::CUB_OP_DUAL_ARG_REDUCE, reduce_params, stream};
    tmp.ExecDualArgReduce(a1_out, aidx1_out, a2_out, aidx2_out, a, stream);
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
 * @param exec
 *   CUDA executor
 */
template <typename OutputTensor, typename InputOperator>
void sort_impl(OutputTensor &a_out, const InputOperator &a,
          const SortDirection_t dir,
          cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  using a_type = typename InputOperator::value_type;
  a_type *out_ptr = nullptr;
  detail::tensor_impl_t<a_type, InputOperator::Rank()> tmp_in;

  // sorting currently requires a contiguous tensor view, so allocate a temporary
  // tensor to copy the input if necessary.
  bool done = false;
  if constexpr (is_tensor_view_v<InputOperator>) {
    if (a.IsContiguous()) {
      make_tensor(tmp_in, a.Data(), a.Shape());
      done = true;
    }
  }

  if (!done) {
    matxAlloc((void**)&out_ptr, TotalSize(a) * sizeof(a_type), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
    make_tensor(tmp_in, out_ptr, a.Shape());
    (tmp_in = a).run(exec);
  }

  detail::sort_impl_inner(a_out, tmp_in, dir, exec);

  if (!done) {
    // We need to free the temporary memory allocated above if we had to make a copy
    matxFree(out_ptr, exec.getStream());
  }
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
 * @param idx_out
 *   Indices of sorted tensor
 * @param a
 *   Input operator
 * @param dir
 *   Direction to sort (either SORT_DIR_ASC or SORT_DIR_DESC)
 * @param exec
 *   CUDA executor
 */
template <typename OutputTensor, typename InputOperator>
void argsort_impl(OutputTensor &idx_out, const InputOperator &a,
          const SortDirection_t dir,
          cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  static constexpr int RANK = OutputTensor::Rank();

  using a_type = typename InputOperator::value_type;
  a_type *a_ptr = nullptr;
  a_type *a_out_ptr = nullptr;
  index_t *idx_in_ptr = nullptr;
  detail::tensor_impl_t<a_type, InputOperator::Rank()> tmp_a;
  detail::tensor_impl_t<a_type, InputOperator::Rank()> tmp_a_out;
  detail::tensor_impl_t<index_t, InputOperator::Rank()> tmp_idx_in;

  // sorting currently requires a contiguous tensor view, so allocate a temporary
  // tensor to copy the input if necessary.
  bool use_a = false;
  if constexpr (is_tensor_view_v<InputOperator>) {
    if (a.IsContiguous()) {
      make_tensor(tmp_a, a.Data(), a.Shape());
      use_a = true;
    }
  }
  if (!use_a) {
    matxAlloc((void**)&a_ptr, TotalSize(a) * sizeof(a_type), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
    make_tensor(tmp_a, a_ptr, a.Shape());
    (tmp_a = a).run(exec);
  }

  // also requires a temporary for output values and input indices
  matxAlloc((void**)&a_out_ptr, TotalSize(a) * sizeof(a_type), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
  make_tensor(tmp_a_out, a_out_ptr, a.Shape());

  matxAlloc((void**)&idx_in_ptr, TotalSize(idx_out) * sizeof(index_t), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
  make_tensor(tmp_idx_in, idx_in_ptr, idx_out.Shape());
  (tmp_idx_in = range<RANK-1>(idx_out.Shape(), 0, 1)).run(exec);

  detail::sort_pairs_impl_inner(idx_out, tmp_idx_in, tmp_a_out, tmp_a, dir, exec);

  if (!use_a) {
    // We need to free the temporary memory allocated above if we had to make a copy
    matxFree(a_ptr, exec.getStream());
  }
  matxFree(a_out_ptr, exec.getStream());
  matxFree(idx_in_ptr, exec.getStream());
#endif
}

template <typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void argsort_impl(OutputTensor &idx_out, const InputOperator &a,
          const SortDirection_t dir,
          [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  static constexpr int RANK = OutputTensor::Rank();
  (idx_out = range<RANK-1>(idx_out.Shape(), 0, 1)).run(exec);
  typename detail::base_type_t<OutputTensor>  out_base = idx_out;
  auto lout = matx::RandomOperatorOutputIterator{out_base};

  if constexpr (RANK == 1) {
    if (dir == SORT_DIR_ASC) {
      std::sort(
          lout, lout + idx_out.Size(0),
          [&a](index_t i, index_t j) { return a(i) < a(j); });
    }
    else {
      std::sort(
          lout, lout + idx_out.Size(0),
          [&a](index_t i, index_t j) { return a(i) > a(j); });
    }
  }
  else if constexpr (RANK == 2) {
    for (index_t b = 0; b < lout.Size(0); b++) {
      if (dir == SORT_DIR_ASC) {
        std::sort( lout + b*a.Size(1), lout + (b+1)*a.Size(1),
                  [&a, b](index_t i, index_t j) { return a(b,i) < a(b,j); });
      }
      else {
        std::sort( lout + b*a.Size(1), lout + (b+1)*a.Size(1),
                  [&a, b](index_t i, index_t j) { return a(b,i) > a(b,j); });
      }
    }
  }
  else {
    MATX_ASSERT_STR(false, matxInvalidDim, "Only 1 and 2D argsort supported on host");
  }
}

template <typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void sort_impl(OutputTensor &a_out, const InputOperator &a,
          const SortDirection_t dir,
          [[maybe_unused]] const HostExecutor<MODE> &exec)
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
                              std::greater<typename InputOperator::value_type>());
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
                                std::greater<typename InputOperator::value_type>());
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
 * @param exec
 *   Executor
 */
template <typename OutputTensor, typename InputOperator>
void cumsum_impl(OutputTensor &a_out, const InputOperator &a,
            cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();

#ifndef MATX_DISABLE_CUB_CACHE
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>::GetCubParams(a_out, a, stream);

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM, detail::EmptyParams_t>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB cumsum transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, detail::EmptyParams_t{}, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecPrefixScanEx(a_out, a, stream);
      },
      exec
    );
#else
  auto tmp =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>{a_out, a, {}, stream};
  tmp.ExecPrefixScanEx(a_out, a, stream);
#endif
#endif
}

template <typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void cumsum_impl(OutputTensor &a_out, const InputOperator &a,
            [[maybe_unused]] const HostExecutor<MODE> &exec)
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
 * bounds and the number of bins.
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
 * @param num_levels
 *   Number of levels
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputOperator>
void hist_impl(OutputTensor &a_out, const InputOperator &a,
          const typename InputOperator::value_type lower,
          const typename InputOperator::value_type upper,
          int num_levels,
          const cudaStream_t stream = 0)
{
  static_assert(std::is_same_v<typename OutputTensor::value_type, int>, "Output histogram operator must use int type");
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  detail::HistEvenParams_t<typename InputOperator::value_type> hp{lower, upper, num_levels};
#ifndef MATX_DISABLE_CUB_CACHE
  using param_type = typename detail::HistEvenParams_t<typename InputOperator::value_type>;
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_HIST_EVEN>::GetCubParams(a_out, a, stream);

  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_HIST_EVEN, param_type>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB histogram transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, hp, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecHistEven(a_out, a, lower, upper, num_levels, stream);
      }
    );

#else
  auto tmp = detail::matxCubPlan_t< OutputTensor,
                                        InputOperator,
                                        detail::CUB_OP_HIST_EVEN,
                                        detail::HistEvenParams_t<typename InputOperator::value_type>>{
      a_out, a, detail::HistEvenParams_t<typename InputOperator::value_type>{hp}, stream};

  tmp.ExecHistEven(a_out, a, lower, upper, num_levels, stream);
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
void find_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};
  cudaStream_t stream = exec.getStream();

#ifndef MATX_DISABLE_CUB_CACHE
  using param_type = typename detail::SelectParams_t<SelectType, CountTensor>;
  // Get cache or new Sort plan if it doesn't exist
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_SELECT_VALS,
                            param_type>::GetCubParams(a_out, a, stream);
  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_VALS, param_type>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB find values transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, cparams, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecSelect(a_out, a, stream);
      },
      exec
    );

#else
  auto tmp = detail::matxCubPlan_t< OutputTensor,
                                        InputOperator,
                                        detail::CUB_OP_SELECT_VALS,
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
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void find_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");
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
void find_idx_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};

#ifndef MATX_DISABLE_CUB_CACHE
  using param_type = typename detail::SelectParams_t<SelectType, CountTensor>;
  // Get cache or new Sort plan if it doesn't exist
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_SELECT_IDX,
                            param_type>::GetCubParams(a_out, a, stream);
  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_IDX, param_type>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB find indices transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, cparams, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecSelectIndex(a_out, a, stream);
      },
      exec
    );

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
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void find_idx_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");
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
 * @param exec
 *   CUDA executor
 */
template <typename CountTensor, typename OutputTensor, typename InputOperator>
void unique_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a,  cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();

  // Allocate space for sorted input since CUB doesn't do unique over unsorted inputs
  auto sort_tensor = make_tensor<typename InputOperator::value_type>(a.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);

  matx::sort_impl(sort_tensor, a, SORT_DIR_ASC, stream);

  auto cparams = detail::UniqueParams_t<CountTensor>{num_found};

#ifndef MATX_DISABLE_CUB_CACHE
  using param_type = typename detail::UniqueParams_t<CountTensor>;
  // Get cache or new Sort plan if it doesn't exist
  auto params =
      detail::matxCubPlan_t<OutputTensor,
                            InputOperator,
                            detail::CUB_OP_UNIQUE,
                            param_type>::GetCubParams(a_out, a, stream);
  using cache_val_type = detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_UNIQUE, param_type>;
  auto cache_id = detail::GetCacheIdFromType<detail::cub_cache_t>();
  MATX_LOG_DEBUG("CUB unique transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::cub_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(a_out, a, cparams, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->ExecUnique(a_out, sort_tensor, stream);
      },
      exec
    );
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
template <typename CountTensor, typename OutputTensor, typename InputOperator, ThreadsMode MODE>
void unique_impl(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
#ifdef __CUDACC__
  static_assert(CountTensor::Rank() == 0, "Num found output tensor rank must be 0");
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
