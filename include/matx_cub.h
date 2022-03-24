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

#include "matx_error.h"
#include "matx_tensor.h"
#include <functional>
#include <cstdio>
#ifdef __CUDACC__  
#include <cub/cub.cuh>
#endif
#include <numeric>

namespace matx {

using namespace std::placeholders;

/**
 * @brief Direction for sorting
 * 
 */
typedef enum { SORT_DIR_ASC, SORT_DIR_DESC } SortDirection_t; 

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
  std::array<index_t, 2> size = {0};
  index_t batches;
  void *a_out;
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

    if constexpr ((RANK == 1 && ( op == CUB_OP_REDUCE_SUM ||
                                  op == CUB_OP_REDUCE ||
                                  op == CUB_OP_REDUCE_MIN ||
                                  op == CUB_OP_REDUCE_MAX)) ||
                  (RANK == 2 && op == CUB_OP_RADIX_SORT)) {
      matxAlloc((void **)&d_offsets, (a.Size(a.Rank() - 2) + 1) * sizeof(index_t),
                MATX_ASYNC_DEVICE_MEMORY, stream);
      for (index_t i = 0; i < a.Size(a.Rank() - 2) + 1; i++) {
        offsets.push_back(i * a.Size(a.Rank() - 1));
      }

      cudaMemcpyAsync(d_offsets, offsets.data(),
                      offsets.size() * sizeof(index_t),
                      cudaMemcpyHostToDevice, stream);
    }

    if constexpr (op == CUB_OP_RADIX_SORT) {
      ExecSort(a_out, a, stream, cparams_.dir);
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

  static CubParams_t GetCubParams(OutputTensor &a_out,
                                  const InputOperator &a)
  {
    CubParams_t params;
    for (int r = a.Rank() - 1; r >= RANK - 2; r--) {
      params.size[1 - r] = a.Size(r);
    }

    params.op = op;
    if constexpr (op == CUB_OP_RADIX_SORT) {
      params.batches = (RANK == 1) ? 1 : a.Size(RANK - 2);
    }
    else if constexpr (op == CUB_OP_INC_SUM || op == CUB_OP_HIST_EVEN) {
      params.batches = a.TotalSize() / a.Size(a.Rank() - 1);
    }
    params.a_out = a_out.Data();
    params.dtype = TypeToInt<T1>();

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
    matxFree(d_offsets);
  }

  template <typename Func>
  void RunBatches(OutputTensor &a_out, const InputOperator &a, const Func &f, int batch_offset)
  {
    using shape_type = index_t;
    size_t total_iter = 1;
    for (int i = 0; i < InputOperator::Rank() - batch_offset; i++) {
      total_iter *= a.Size(i);
    }

    // Get total indices per batch
    size_t total_per_batch = 1;
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
    const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
    if (RANK == 1 || d_temp == nullptr) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        if (a.IsContiguous()) {      
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a_out.Lsize() + 1), lower, upper,
              static_cast<int>(a.Lsize()), stream);
        }
        else {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, RandomOperatorIterator{base}, a_out.Data(),
              static_cast<int>(a_out.Lsize() + 1), lower, upper,
              static_cast<int>(a.Lsize()), stream);          
        }
      }
      else {
        cub::DeviceHistogram::HistogramEven(
            d_temp, temp_storage_bytes, RandomOperatorIterator{a}, a_out.Data(),
            static_cast<int>(a_out.Lsize() + 1), lower, upper,
            static_cast<int>(a.Lsize()), stream);               
      }
    }
    else { // Batch higher dims
      auto ft = [&](auto ...p){ return cub::DeviceHistogram::HistogramEven(p...); };
      auto f = std::bind(ft, d_temp, temp_storage_bytes, _1, _2, static_cast<int>(a_out.Lsize() + 1), lower, upper,
            static_cast<int>(a.Lsize()), stream);
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
    if (RANK == 1 || d_temp == nullptr) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {          
          cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, a.Data(),
                                        a_out.Data(), static_cast<int>(a.Lsize()),
                                        stream);
        }
        else {
          cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, RandomOperatorIterator{base},
                                        a_out.Data(), static_cast<int>(a.Lsize()),
                                        stream);          
        }
      }
      else {
        cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, RandomOperatorIterator{a},
                                      a_out.Data(), static_cast<int>(a.Lsize()),
                                      stream);          
      }
    }
    else {
        auto ft = [&](auto ...p){ cub::DeviceScan::InclusiveSum(p...); };
        auto f = std::bind(ft, d_temp, temp_storage_bytes, _1, _2, static_cast<int>(a.Lsize()), stream);
        RunBatches(a_out, a, f, 1);
    }
#endif    
  }

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
                       const cudaStream_t stream,
                       const SortDirection_t dir = SORT_DIR_ASC)
  {
#ifdef __CUDACC__
    static_assert(is_tensor_view_v<InputOperator>, "Sorting only accepts tensors for now (no operators)");
    MATX_ASSERT_STR(a.IsContiguous(), matxInvalidType, "Tensor must be contiguous in memory for sorting");

    if constexpr (is_tensor_view_v<InputOperator>) {
      if (RANK == 1) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;     
        if (dir == SORT_DIR_ASC) {
          cub::DeviceRadixSort::SortKeys(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a.Lsize()), 0, sizeof(T1) * 8, stream);
        }
        else {
          cub::DeviceRadixSort::SortKeysDescending(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a.Lsize()), 0, sizeof(T1) * 8, stream);
        }
      }        
      else if (RANK == 2 || d_temp == nullptr) {              
        if (dir == SORT_DIR_ASC) {
          cub::DeviceSegmentedRadixSort::SortKeys(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
              d_offsets, d_offsets + 1, 0, sizeof(T1) * 8, stream);
        }
        else {
          cub::DeviceSegmentedRadixSort::SortKeysDescending(
              d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
              static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
              d_offsets, d_offsets + 1, 0, sizeof(T1) * 8, stream);
        }
      }
      else {
        constexpr int batch_offset = 2;
        using shape_type = index_t;
        size_t total_iter = 1;
        for (int i = 0; i < InputOperator::Rank() - batch_offset; i++) {
          total_iter *= a.Size(i);
        }

        std::array<shape_type, InputOperator::Rank()> idx{0};

        if (dir == SORT_DIR_ASC) {
          auto ft = [&](auto ...p){ cub::DeviceSegmentedRadixSort::SortKeys(p...); };

          auto f = std::bind(ft, d_temp, temp_storage_bytes, _1, _2, static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
              d_offsets, d_offsets + 1, static_cast<int>(0), static_cast<int>(sizeof(T1) * 8), stream, false);

          for (size_t iter = 0; iter < total_iter; iter++) {
            auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
            auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

            f(ap, aop);

            // Update all but the last batch_offset indices
            UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
          }
        }
        else {
          auto ft = [&](auto ...p){ cub::DeviceSegmentedRadixSort::SortKeysDescending(p...); };
          auto f = std::bind(ft, d_temp, temp_storage_bytes, _1, _2, static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
              d_offsets, d_offsets + 1, static_cast<int>(0), static_cast<int>(sizeof(T1) * 8), stream, false);

          for (size_t iter = 0; iter < total_iter; iter++) {
            auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
            auto aop = std::apply([&a_out](auto... param) { return a_out.GetPointer(param...); }, idx);

            f(ap, aop);

            // Update all but the last batch_offset indices
            UpdateIndices<InputOperator, shape_type, InputOperator::Rank()>(a, idx, batch_offset);
          }        
        }
      }
    }
#endif    
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
    if constexpr (RANK == 0) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {   
          cub::DeviceReduce::Reduce(d_temp, 
                                    temp_storage_bytes, 
                                    a.Data(), 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)), 
                                    cparams_.reduce_op, 
                                    cparams_.init,
                                    stream); 
        }
        else {
          cub::DeviceReduce::Reduce(d_temp, 
                                    temp_storage_bytes, 
                                    RandomOperatorIterator{base}, 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)), 
                                    cparams_.reduce_op, 
                                    cparams_.init,
                                    stream);            
        }
      }
      else {
        cub::DeviceReduce::Reduce(d_temp, 
                                  temp_storage_bytes, 
                                  RandomOperatorIterator{a}, 
                                  a_out.Data(), 
                                  static_cast<int>(TotalSize(a)), 
                                  cparams_.reduce_op, 
                                  cparams_.init,
                                  stream);              
      }                                     
    }
    else if constexpr (RANK == 1) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) { 
          cub::DeviceSegmentedReduce::Reduce( d_temp, 
                                              temp_storage_bytes, 
                                              a.Data(), 
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);      
        }
        else {
          cub::DeviceSegmentedReduce::Reduce( d_temp, 
                                              temp_storage_bytes, 
                                              RandomOperatorIterator{base}, 
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);             
        }
      }
      else {
        cub::DeviceSegmentedReduce::Reduce( d_temp, 
                                            temp_storage_bytes, 
                                            RandomOperatorIterator{a},  
                                            a_out.Data(), 
                                            static_cast<int>(a_out.Size(0)), 
                                            d_offsets, 
                                            d_offsets + 1, 
                                            stream);         
      }        
    }
#endif    
  }  

  /**
   * Execute a sum on a tensor
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
    if constexpr (RANK == 0) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceReduce::Sum(d_temp, 
                                    temp_storage_bytes, 
                                    a.Data(), 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)),
                                    stream);           
        }
        else {
          cub::DeviceReduce::Sum(d_temp, 
                                    temp_storage_bytes, 
                                    RandomOperatorIterator{base}, 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)),
                                    stream);  
        }
      }
      else {
        cub::DeviceReduce::Sum(d_temp, 
                                  temp_storage_bytes, 
                                  RandomOperatorIterator{a}, 
                                  a_out.Data(), 
                                  static_cast<int>(TotalSize(a)),
                                  stream);              
      }                                     
    }
    else if constexpr (RANK == 1) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceSegmentedReduce::Sum( d_temp, 
                                              temp_storage_bytes, 
                                              a.Data(), 
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream); 
        }     
        else {
          cub::DeviceSegmentedReduce::Sum( d_temp, 
                                              temp_storage_bytes, 
                                              RandomOperatorIterator{base},  
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);            
        }
      }
      else {
        cub::DeviceSegmentedReduce::Sum( d_temp, 
                                            temp_storage_bytes, 
                                            RandomOperatorIterator{a},  
                                            a_out.Data(), 
                                            static_cast<int>(a_out.Size(0)), 
                                            d_offsets, 
                                            d_offsets + 1, 
                                            stream);         
      }        
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
    if constexpr (RANK == 0) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceReduce::Min(d_temp, 
                                    temp_storage_bytes, 
                                    a.Data(), 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)),
                                    stream); 
        }
        else {
          cub::DeviceReduce::Min(d_temp, 
                                    temp_storage_bytes, 
                                    RandomOperatorIterator{base}, 
                                    a_out.Data(), 
                                    static_cast<int>(TotalSize(a)),
                                    stream);           
        }
      }
      else {
        cub::DeviceReduce::Min(d_temp, 
                                  temp_storage_bytes, 
                                  RandomOperatorIterator{a}, 
                                  a_out.Data(), 
                                  static_cast<int>(TotalSize(a)),
                                  stream);              
      }                                     
    }
    else if constexpr (RANK == 1) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceSegmentedReduce::Min( d_temp, 
                                              temp_storage_bytes, 
                                              a.Data(), 
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);      
        }
        else {
          cub::DeviceSegmentedReduce::Min( d_temp, 
                                              temp_storage_bytes, 
                                              RandomOperatorIterator{base},  
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);            
        }
      }
      else {
        cub::DeviceSegmentedReduce::Min( d_temp, 
                                            temp_storage_bytes, 
                                            RandomOperatorIterator{a},  
                                            a_out.Data(), 
                                            static_cast<int>(a_out.Size(0)), 
                                            d_offsets, 
                                            d_offsets + 1, 
                                            stream);         
      }        
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
    if constexpr (RANK == 0) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceReduce::Max(d_temp, 
                                    temp_storage_bytes, 
                                    a.Data(), 
                                    a_out.Data(), 
                                    static_cast<int>(a.TotalSize()),
                                    stream); 
        }
        else {
          cub::DeviceReduce::Max(d_temp, 
                                    temp_storage_bytes, 
                                    RandomOperatorIterator{base}, 
                                    a_out.Data(), 
                                    static_cast<int>(a.TotalSize()),
                                    stream);           
        }
      }
      else {
        cub::DeviceReduce::Max(d_temp, 
                                  temp_storage_bytes, 
                                  RandomOperatorIterator{a}, 
                                  a_out.Data(), 
                                  static_cast<int>(a.TotalSize()),
                                  stream);              
      }                                     
    }
    else if constexpr (RANK == 1) {
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceSegmentedReduce::Max( d_temp, 
                                              temp_storage_bytes, 
                                              a.Data(), 
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);
        }
        else {
          cub::DeviceSegmentedReduce::Max( d_temp, 
                                              temp_storage_bytes, 
                                              RandomOperatorIterator{base},  
                                              a_out.Data(), 
                                              static_cast<int>(a_out.Size(0)), 
                                              d_offsets, 
                                              d_offsets + 1, 
                                              stream);             
        }
      }
      else {
        cub::DeviceSegmentedReduce::Max( d_temp, 
                                            temp_storage_bytes, 
                                            RandomOperatorIterator{a},  
                                            a_out.Data(), 
                                            static_cast<int>(a_out.Size(0)), 
                                            d_offsets, 
                                            d_offsets + 1, 
                                            stream);         
      }        
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
    if constexpr (is_tensor_view_v<InputOperator>) {
      const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
      if (a.IsContiguous()) {
        cub::DeviceSelect::If(d_temp, 
                              temp_storage_bytes, 
                              a.Data(), 
                              a_out.Data(), 
                              cparams_.num_found.Data(),
                              static_cast<int>(a.TotalSize()),
                              cparams_.op,
                              stream); 
      }
      else {
        cub::DeviceSelect::If(d_temp, 
                              temp_storage_bytes, 
                              RandomOperatorIterator{base},
                              a_out.Data(), 
                              cparams_.num_found.Data(),
                              static_cast<int>(a.TotalSize()),
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
                            static_cast<int>(a.TotalSize()),
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
    if constexpr (is_tensor_view_v<InputOperator>) {
      if (a.IsContiguous()) {
        cub::DeviceSelect::If(d_temp, 
                              temp_storage_bytes, 
                              cub::CountingInputIterator<index_t>{0}, 
                              a_out.Data(), 
                              cparams_.num_found.Data(),
                              static_cast<int>(a.TotalSize()),
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
                              static_cast<int>(a.TotalSize()),
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
                            static_cast<int>(a.TotalSize()),
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
      if constexpr (is_tensor_view_v<InputOperator>) {
        const tensor_impl_t<typename InputOperator::scalar_type, InputOperator::Rank(), typename InputOperator::desc_type> base = a;
        if (a.IsContiguous()) {
          cub::DeviceSelect::Unique(d_temp, 
                                temp_storage_bytes, 
                                a.Data(), 
                                a_out.Data(), 
                                cparams_.num_found.Data(),
                                static_cast<int>(a.TotalSize()),
                                stream); 
        }
        else {
          cub::DeviceSelect::Unique(d_temp, 
                                temp_storage_bytes, 
                                RandomOperatorIterator{base},
                                a_out.Data(), 
                                cparams_.num_found.Data(),
                                static_cast<int>(a.TotalSize()),
                                stream);              
        }
    }
    else {
      cub::DeviceSelect::Unique(d_temp, 
                            temp_storage_bytes, 
                            RandomOperatorIterator{a},
                            a_out.Data(), 
                            cparams_.num_found.Data(),
                            static_cast<int>(a.TotalSize()),
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
  std::vector<index_t> offsets;
  index_t *d_offsets = nullptr;
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
    return (std::hash<uint64_t>()(k.batches)) +
           (std::hash<uint64_t>()((uint64_t)k.size[0])) +
           (std::hash<uint64_t>()((uint64_t)k.size[1])) +
           (std::hash<uint64_t>()((uint64_t)k.a_out)) +
           (std::hash<uint64_t>()((uint64_t)k.stream)) +
           (std::hash<uint64_t>()((uint64_t)k.op));
  }
};

/**
 * Test Sort parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct CubParamsKeyEq {
  bool operator()(const CubParams_t &l, const CubParams_t &t) const noexcept
  {
    return l.size[0] == t.size[0] && l.size[1] == t.size[1] && l.a_out == t.a_out &&
           l.batches == t.batches && l.dtype == t.dtype &&
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
  // Get parameters required by these tensors
  using param_type = typename detail::ReduceParams_t<ReduceOp, typename InputOperator::scalar_type>;
  auto reduce_params = param_type{ReduceOp{}, init};
  auto params =
      detail::matxCubPlan_t<OutputTensor, 
                            InputOperator, 
                            detail::CUB_OP_REDUCE, 
                            param_type>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type>{
        a_out, a, reduce_params, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecReduce(a_out, a, stream);
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE, param_type> *>(
            ret.value());
    type->ExecReduce(a_out, a, stream);
  }
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
  auto params =
      detail::matxCubPlan_t<OutputTensor, 
                            InputOperator, 
                            detail::CUB_OP_REDUCE_SUM>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM>{
        a_out, a, {}, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSum(a_out, a, stream);
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_SUM> *>(
            ret.value());
    type->ExecSum(a_out, a, stream);
  }
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
  auto params =
      detail::matxCubPlan_t<OutputTensor, 
                            InputOperator, 
                            detail::CUB_OP_REDUCE_MIN>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN>{
        a_out, a, {}, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecMin(a_out, a, stream);
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MIN> *>(
            ret.value());
    type->ExecMin(a_out, a, stream);
  }
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
  auto params =
      detail::matxCubPlan_t<OutputTensor, 
                            InputOperator, 
                            detail::CUB_OP_REDUCE_MAX>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX>{
        a_out, a, {}, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecMax(a_out, a, stream);
  }
  else {
    auto type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_REDUCE_MAX> *>(
            ret.value());
    type->ExecMax(a_out, a, stream);
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
          const SortDirection_t dir = SORT_DIR_ASC,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT>::GetCubParams(a_out, a);
  params.stream = stream;

  detail::SortParams_t p{dir};

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)>{
        a_out, a, p, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSort(a_out, a, stream, dir);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_RADIX_SORT, decltype(p)> *>(
            ret.value());
    sort_type->ExecSort(a_out, a, stream, dir);
  }
#endif  
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
            const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp =
        new detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM>{a_out, a, {}, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecPrefixScanEx(a_out, a, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_INC_SUM> *>(ret.value());
    sort_type->ExecPrefixScanEx(a_out, a, stream);
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
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_HIST_EVEN>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    detail::HistEvenParams_t<typename InputOperator::scalar_type> hp{lower, upper};
    auto tmp = new detail::matxCubPlan_t< OutputTensor, 
                                          InputOperator, 
                                          detail::CUB_OP_HIST_EVEN, 
                                          detail::HistEvenParams_t<typename InputOperator::scalar_type>>{
        a_out, a, detail::HistEvenParams_t<typename InputOperator::scalar_type>{hp}, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecHistEven(a_out, a, lower, upper, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, 
            detail::CUB_OP_HIST_EVEN, detail::HistEvenParams_t<typename InputOperator::scalar_type>> *>(
            ret.value());
    sort_type->ExecHistEven(a_out, a, lower, upper, stream);
  }
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
 * @param stream
 *   CUDA stream
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");

  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT, SelectType>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};

  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor, 
                                          InputOperator, 
                                          detail::CUB_OP_SELECT, 
                                          decltype(cparams)>{a_out, a, cparams, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSelect(a_out, a, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecSelect(a_out, a, stream);
  }
#endif  
}

/**
 * Reduce indices that meet a certain criteria
 *
 * Finds all indices of values meeting the criteria specified in SelectOp, and saves them out to an output tensor. This
 * function is different from the MatX IF operator in that this performs a reduction on the input, whereas IF
 * is only for element-wise output. Output tensor must be large enough to hold unique entries. To be safe, 
 * this can be the same size as the input, but if something is known about the data to indicate not as many 
 * entries are needed, the output can be smaller. This 
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
 * @param stream
 *   CUDA stream
 */
template <typename SelectType, typename CountTensor, typename OutputTensor, typename InputOperator>
void find_idx(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");

  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_IDX, SelectType>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);
  auto cparams = detail::SelectParams_t<SelectType, CountTensor>{sel, num_found};

  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor, 
                                          InputOperator, 
                                          detail::CUB_OP_SELECT_IDX, 
                                          decltype(cparams)>{a_out, a, cparams, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSelectIndex(a_out, a, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_SELECT_IDX, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecSelectIndex(a_out, a, stream);
  }
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
 * @param stream
 *   CUDA stream
 */
template <typename CountTensor, typename OutputTensor, typename InputOperator>
void unique(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a,  const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  static_assert(num_found.Rank() == 0, "Num found output tensor rank must be 0");

  // Allocate space for sorted input since CUB doesn't do unique over unsorted inputs
  typename InputOperator::scalar_type *sort_ptr;
  matxAlloc((void **)&sort_ptr, a.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto sort_tensor = make_tensor<typename InputOperator::scalar_type>(sort_ptr, a.Shape());

  matx::sort(sort_tensor, a, SORT_DIR_ASC, stream);

  auto cparams = detail::UniqueParams_t<CountTensor>{num_found};
  // Get parameters required by these tensors
  auto params =
      detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_UNIQUE, decltype(cparams)>::GetCubParams(a_out, sort_tensor);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = detail::cub_cache.Lookup(params);


  if (ret == std::nullopt) {
    auto tmp = new detail::matxCubPlan_t< OutputTensor, 
                                          InputOperator, 
                                          detail::CUB_OP_UNIQUE, 
                                          decltype(cparams)>{a_out, sort_tensor, cparams, stream};
    detail::cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecUnique(a_out, sort_tensor, stream);
  }
  else {
    auto sort_type =
        static_cast<detail::matxCubPlan_t<OutputTensor, InputOperator, detail::CUB_OP_UNIQUE, decltype(cparams)> *>(
            ret.value());
    sort_type->ExecUnique(a_out, sort_tensor, stream);
  }
#endif  
}
}; // namespace matx