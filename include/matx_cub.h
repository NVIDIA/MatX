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

#include "matx_dim.h"
#include "matx_error.h"
#include "matx_tensor.h"
#include <any>
#include <cstdio>
#ifdef __CUDACC__  
#include <cub/cub.cuh>
#endif
#include <numeric>

namespace matx {

/**
 * Parameters needed to execute a sort operation.
 */

typedef enum {
  CUB_OP_RADIX_SORT,
  CUB_OP_INC_SUM,
  CUB_OP_HIST_EVEN
} CUBOperation_t;

struct CubParams_t {
  CUBOperation_t op;
  index_t size;
  index_t batches;
  void *A;
  void *a_out;
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename T> struct HistEvenParams_t {
  T lower_level;
  T upper_level;
};

typedef enum { SORT_DIR_ASC, SORT_DIR_DESC } SortDirection_t;

template <typename OutputTensor, typename InputTensor, CUBOperation_t op>
class matxCubPlan_t {
  static_assert(OutputTensor::Rank() == InputTensor::Rank(), "CUB input and output tensor ranks must match");
  static constexpr int RANK = OutputTensor::Rank();
  static_assert(RANK >= 1);
  using T1 = typename InputTensor::scalar_type;
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
   * @tparam T1
   *   Data type of A matrix
   * @tparam RANK
   *   Rank of A matrix
   * @tparam op
   *   CUB operation to perform
   *
   * @param a
   *   Input tensor view
   * @param a_out
   *   Sorted output
   * @param params
   *   Parameter structure specific to the operation
   *
   */
  matxCubPlan_t(OutputTensor &a_out, const InputTensor &a,
                const std::any &cparams, const cudaStream_t stream = 0)
  {
#ifdef __CUDACC__  
    // Input/output tensors much match rank/dims
    if constexpr (op != CUB_OP_HIST_EVEN) {
      for (int i = 0; i < a.Rank(); i++) {
        MATX_ASSERT(a.Size(i) == a_out.Size(i), matxInvalidSize);
      }
    }

    if constexpr (op == CUB_OP_RADIX_SORT) {
      // Create temporary allocation space for sorting. Only contiguous for now.
      // The memory required should be the same for ascending or descending, so
      // just use ascending here.
      if constexpr (RANK == 1) {
        cub::DeviceRadixSort::SortKeys(
            NULL, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Lsize()), 0, sizeof(T1) * 8, stream);
      }
      else {
        matxAlloc((void **)&d_offsets, (a.Size(RANK - 2) + 1) * sizeof(index_t),
                  MATX_ASYNC_DEVICE_MEMORY, stream);
        for (index_t i = 0; i < a.Size(RANK - 2) + 1; i++) {
          offsets.push_back(i * a.Lsize());
        }

        cudaMemcpyAsync(d_offsets, offsets.data(),
                        offsets.size() * sizeof(index_t),
                        cudaMemcpyHostToDevice, stream);

        cub::DeviceSegmentedRadixSort::SortKeys(
            NULL, temp_storage_bytes, a.Data(), a_out.Data(),
            static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
            d_offsets, d_offsets + 1, 0, sizeof(T1) * 8, stream);
      }
    }
    else if constexpr (op == CUB_OP_INC_SUM) {
      // Scan only is capable of non-batched mode at the moment
      cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, a.Data(),
                                    a_out.Data(), static_cast<int>(a.Lsize()),
                                    stream);
    }
    else if constexpr (op == CUB_OP_HIST_EVEN) {
      HistEvenParams_t<T1> p = std::any_cast<HistEvenParams_t<T1>>(cparams);
      cub::DeviceHistogram::HistogramEven(
          NULL, temp_storage_bytes, a.Data(), a_out.Data(),
          static_cast<int>(a_out.Lsize() + 1), p.lower_level, p.upper_level,
          static_cast<int>(a.Lsize()),
          stream); // Remove this case once CUB is fixed
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
                                  const InputTensor &a)
  {
    CubParams_t params;
    params.size = a.Lsize();
    params.A = a.Data();
    params.op = op;
    if constexpr (op == CUB_OP_RADIX_SORT) {
      params.batches = (RANK == 1) ? 1 : a.Size(RANK - 2);
    }
    else if constexpr (op == CUB_OP_INC_SUM || op == CUB_OP_HIST_EVEN) {
      params.batches = a.TotalSize() / a.Lsize();
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
                           const InputTensor &a, const T1 lower,
                           const T1 upper, const cudaStream_t stream)
  {
#ifdef __CUDACC__      
    if constexpr (RANK == 1) {
      cub::DeviceHistogram::HistogramEven(
          d_temp, temp_storage_bytes, a.Data(), a_out.Data(),
          static_cast<int>(a_out.Lsize() + 1), lower, upper,
          static_cast<int>(a.Lsize()), stream);
    }
    else if constexpr (RANK == 2) {
      for (index_t i = 0; i < a.Size(0); i++) {
        cub::DeviceHistogram::HistogramEven(
            d_temp, temp_storage_bytes, &a(i, 0), &a_out(i, 0),
            static_cast<int>(a_out.Lsize() + 1), lower, upper,
            static_cast<int>(a.Lsize()), stream);
      }
    }
    else if constexpr (RANK == 3) {
      for (index_t i = 0; i < a.Size(0); i++) {
        for (index_t j = 0; j < a.Size(1); j++) {
          cub::DeviceHistogram::HistogramEven(
              d_temp, temp_storage_bytes, &a(i, j, 0), &a_out(i, j, 0),
              static_cast<int>(a_out.Lsize() + 1), lower, upper,
              static_cast<int>(a.Lsize()), stream);
        }
      }
    }
    else if constexpr (RANK == 4) {
      for (index_t i = 0; i < a.Size(0); i++) {
        for (index_t j = 0; j < a.Size(1); j++) {
          for (index_t k = 0; k < a.Size(2); k++) {
            cub::DeviceHistogram::HistogramEven(
                d_temp, temp_storage_bytes, &a(i, j, k, 0), &a_out(i, j, k, 0),
                static_cast<int>(a_out.Lsize() + 1), lower, upper,
                static_cast<int>(a.Lsize()), stream);
          }
        }
      }
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
                               const InputTensor &a,
                               const cudaStream_t stream)
  {
#ifdef __CUDACC__      
    if constexpr (RANK == 1) {
      cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, a.Data(),
                                    a_out.Data(), static_cast<int>(a.Lsize()),
                                    stream);
    }
    else if constexpr (RANK == 2) {
      for (index_t i = 0; i < a.Size(0); i++) {
        cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, &a(i, 0),
                                      &a_out(i, 0), static_cast<int>(a.Lsize()),
                                      stream);
      }
    }
    else if constexpr (RANK == 3) {
      for (index_t i = 0; i < a.Size(0); i++) {
        for (index_t j = 0; j < a.Size(1); j++) {
          cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes, &a(i, j, 0),
                                        &a_out(i, j, 0),
                                        static_cast<int>(a.Lsize()), stream);
        }
      }
    }
    else if constexpr (RANK == 4) {
      for (index_t i = 0; i < a.Size(0); i++) {
        for (index_t j = 0; j < a.Size(1); j++) {
          for (index_t k = 0; k < a.Size(2); k++) {
            cub::DeviceScan::InclusiveSum(d_temp, temp_storage_bytes,
                                          &a(i, j, k, 0), &a_out(i, j, k, 0),
                                          static_cast<int>(a.Lsize()), stream);
          }
        }
      }
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
                       const InputTensor &a,
                       const cudaStream_t stream,
                       const SortDirection_t dir = SORT_DIR_ASC)
  {
#ifdef __CUDACC__      
    if constexpr (RANK == 1) {
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
    else {
      if constexpr (RANK == 2) {
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
      else if constexpr (RANK == 3) {
        for (index_t i = 0; i < a.Size(0); i++) {
          if (dir == SORT_DIR_ASC) {
            cub::DeviceSegmentedRadixSort::SortKeys(
                d_temp, temp_storage_bytes, &a(i, 0, 0), &a_out(i, 0, 0),
                static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
                d_offsets, d_offsets + 1, 0, sizeof(T1) * 8, stream);
          }
          else {
            cub::DeviceSegmentedRadixSort::SortKeysDescending(
                d_temp, temp_storage_bytes, &a(i, 0, 0), &a_out(i, 0, 0),
                static_cast<int>(a.Lsize()), static_cast<int>(a.Size(RANK - 2)),
                d_offsets, d_offsets + 1, 0, sizeof(T1) * 8, stream);
          }
        }
      }
      else if constexpr (RANK == 4) {
        for (index_t i = 0; i < a.Size(0); i++) {
          for (index_t j = 0; j < a.Size(1); j++) {
            if (dir == SORT_DIR_ASC) {
              cub::DeviceSegmentedRadixSort::SortKeys(
                  d_temp, temp_storage_bytes, &a(i, j, 0, 0),
                  &a_out(i, j, 0, 0), static_cast<int>(a.Lsize()),
                  static_cast<int>(a.Size(RANK - 2)), d_offsets, d_offsets + 1,
                  0, sizeof(T1) * 8, stream);
            }
            else {
              cub::DeviceSegmentedRadixSort::SortKeysDescending(
                  d_temp, temp_storage_bytes, &a(i, j, 0, 0),
                  &a_out(i, j, 0, 0), static_cast<int>(a.Lsize()),
                  static_cast<int>(a.Size(RANK - 2)), d_offsets, d_offsets + 1,
                  0, sizeof(T1) * 8, stream);
            }
          }
        }
      }
    }
#endif    
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  CubParams_t params;
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
    return (std::hash<uint64_t>()(k.size)) + (std::hash<uint64_t>()(k.batches)) +
           (std::hash<uint64_t>()((uint64_t)k.A)) +
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
    return l.size == t.size && l.A == t.A && l.a_out == t.a_out &&
           l.batches == t.batches && l.dtype == t.dtype &&
           l.stream == t.stream && l.op == t.op;
  }
};

// Static caches of Sort handles
static matxCache_t<CubParams_t, CubParamsKeyHash, CubParamsKeyEq> cub_cache;

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
template <typename OutputTensor, typename InputTensor>
void sort(OutputTensor &a_out, const InputTensor &a,
          const SortDirection_t dir = SORT_DIR_ASC,
          const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  // Get parameters required by these tensors
  auto params =
      matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_RADIX_SORT>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_RADIX_SORT>{
        a_out, a, {}, stream};
    cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecSort(a_out, a, stream, dir);
  }
  else {
    auto sort_type =
        static_cast<matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_RADIX_SORT> *>(
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
template <typename OutputTensor, typename InputTensor>
void cumsum(OutputTensor &a_out, const InputTensor &a,
            const cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  // Get parameters required by these tensors
  auto params =
      matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_INC_SUM>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp =
        new matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_INC_SUM>{a_out, a, {}, stream};
    cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecPrefixScanEx(a_out, a, stream);
  }
  else {
    auto sort_type =
        static_cast<matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_INC_SUM> *>(ret.value());
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
template <typename OutputTensor, typename InputTensor>
void hist(OutputTensor &a_out, const InputTensor &a,
          const typename InputTensor::scalar_type lower, 
          const typename InputTensor::scalar_type upper, const cudaStream_t stream = 0)
{
  static_assert(std::is_same_v<typename OutputTensor::scalar_type, int>, "Output histogram tensor must use int type");
#ifdef __CUDACC__    
  // Get parameters required by these tensors
  auto params =
      matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_HIST_EVEN>::GetCubParams(a_out, a);
  params.stream = stream;

  // Get cache or new Sort plan if it doesn't exist
  auto ret = cub_cache.Lookup(params);
  if (ret == std::nullopt) {
    HistEvenParams_t<typename InputTensor::scalar_type> hp{lower, upper};
    auto tmp = new matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_HIST_EVEN>{
        a_out, a, std::make_any<HistEvenParams_t<typename InputTensor::scalar_type>>(hp), stream};
    cub_cache.Insert(params, static_cast<void *>(tmp));
    tmp->ExecHistEven(a_out, a, lower, upper, stream);
  }
  else {
    auto sort_type =
        static_cast<matxCubPlan_t<OutputTensor, InputTensor, CUB_OP_HIST_EVEN> *>(
            ret.value());
    sort_type->ExecHistEven(a_out, a, lower, upper, stream);
  }
#endif  
}
}; // namespace matx