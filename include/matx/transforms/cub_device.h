////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#ifdef __CUDACC__

#ifndef __CUDACC_RTC__
  #include "matx/core/capabilities.h"
#else
  #include "matx/core/operator_options.h"
#endif
#include "matx/core/vector.h"
#if defined(MATX_EN_JIT) && !defined(__CUDACC_RTC__)
  #include "matx/core/nvrtc_helper.h"
#endif
#include <cuda/std/version>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#if defined(MATX_EN_JIT)
  #ifndef CCCL_VERSION
    #error "MatX CUB block JIT support requires CCCL 3.3.0 or newer"
  #elif CCCL_VERSION < 3003000
    #error "MatX CUB block JIT support requires CCCL 3.3.0 or newer"
  #endif
#endif
#ifndef __CUDACC_RTC__
  #include <cuda/cmath>
  #include <cuda/std/__algorithm/min.h>
#endif
#include <cuda/std/limits>
#include <cuda/std/utility>

namespace matx {
namespace detail {

  enum class BlockReduceType {
    SUM,
    MIN,
    MAX,
    PRODUCT
  };

  enum class BlockSortDirection {
    ASCENDING,
    DESCENDING
  };

  enum class CubBlockAlgorithm {
    REDUCE,
    SCAN,
    SORT,
    SORT_PAIRS
  };

#ifndef __CUDACC_RTC__
  static constexpr int CubJitMaxBlockThreads = 1024;

  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int MaxCubJitElementsPerThreadByBytes(int limit = 32)
  {
    constexpr int element_size = static_cast<int>(sizeof(T));
    int byte_limited = 16 / element_size;
    if (byte_limited < 1) {
      byte_limited = 1;
    }

    const int capped = cuda::std::min(limit, byte_limited);
    return capped > 0 ? static_cast<int>(cuda::prev_power_of_two(capped)) : 0;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitMaxReductionEPT(index_t reduce_size, int limit)
  {
    if (reduce_size <= 0 || limit <= 0) {
      return 0;
    }

    const auto capped = cuda::std::min(reduce_size, static_cast<index_t>(limit));
    return static_cast<int>(cuda::prev_power_of_two(capped));
  }

  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitMaxReductionEPT(index_t reduce_size)
  {
    return CubJitMaxReductionEPT(reduce_size, MaxCubJitElementsPerThreadByBytes<T>());
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitReductionBlockThreads(index_t reduce_size, int ept)
  {
    if (reduce_size <= 0 || ept <= 0) {
      return 0;
    }

    return static_cast<int>((reduce_size + ept - 1) / ept);
  }

  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool CubJitReductionFitsInBlock(index_t reduce_size, int max_threads = CubJitMaxBlockThreads)
  {
    if (reduce_size <= 0) {
      return false;
    }

    const int max_ept = CubJitMaxReductionEPT<T>(reduce_size);
    const int block_threads = CubJitReductionBlockThreads(reduce_size, max_ept);
    return max_ept > 0 && block_threads > 0 && block_threads <= max_threads;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitMaxPowerOfTwoCollectiveEPT(index_t critical_dim_size, int limit)
  {
    if (critical_dim_size <= 0 || limit <= 0 || !cuda::is_power_of_two(critical_dim_size)) {
      return 0;
    }

    const auto capped = cuda::std::min(critical_dim_size, static_cast<index_t>(limit));
    return static_cast<int>(cuda::prev_power_of_two(capped));
  }

  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitMaxPowerOfTwoCollectiveEPT(index_t critical_dim_size)
  {
    return CubJitMaxPowerOfTwoCollectiveEPT(critical_dim_size, MaxCubJitElementsPerThreadByBytes<T>());
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr int CubJitPowerOfTwoCollectiveBlockThreads(index_t critical_dim_size, int ept)
  {
    if (critical_dim_size <= 0 || ept <= 0 || (critical_dim_size % ept) != 0) {
      return 0;
    }

    const auto block_threads = critical_dim_size / ept;
    return cuda::is_power_of_two(block_threads) ? static_cast<int>(block_threads) : 0;
  }

  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool CubJitPowerOfTwoCollectiveFitsInBlock(index_t critical_dim_size, int max_threads = CubJitMaxBlockThreads)
  {
    if (critical_dim_size <= 0) {
      return false;
    }

    const int max_ept = CubJitMaxPowerOfTwoCollectiveEPT<T>(critical_dim_size);
    const int block_threads = CubJitPowerOfTwoCollectiveBlockThreads(critical_dim_size, max_ept);
    return block_threads > 0 && block_threads <= max_threads;
  }
#endif

  template <typename CapType, typename Op, size_t Rank, size_t... I>
  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) block_op_get_value_at(
      const Op &op, const cuda::std::array<index_t, Rank> &indices, cuda::std::index_sequence<I...>)
  {
    return op.template operator()<CapType>(indices[I]...);
  }

  template <typename CapType, typename Op, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) block_op_get_last_dim_item(const Op &op,
                                                                            int item,
                                                                            Is... indices)
  {
    static_assert(sizeof...(Is) > 0, "Block CUB operators require at least one index");
    cuda::std::array<index_t, sizeof...(Is)> scalar_indices{static_cast<index_t>(indices)...};
    scalar_indices[sizeof...(Is) - 1] =
        static_cast<index_t>(threadIdx.x) * static_cast<int>(CapType::ept) + static_cast<index_t>(item);
    using ScalarCap = typename CapType::scalar_cap;
    auto value = block_op_get_value_at<ScalarCap>(op,
                                                  scalar_indices,
                                                  cuda::std::make_index_sequence<sizeof...(Is)>{});
    return GetVectorVal(value, 0);
  }

  template <typename CapType, typename Op, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) block_op_get_last_dim_vector(const Op &op,
                                                                              Is... indices)
  {
    static_assert(sizeof...(Is) > 0, "Block CUB operators require at least one index");
    cuda::std::array<index_t, sizeof...(Is)> vector_indices{static_cast<index_t>(indices)...};
    vector_indices[sizeof...(Is) - 1] = static_cast<index_t>(threadIdx.x);
    return block_op_get_value_at<CapType>(op,
                                          vector_indices,
                                          cuda::std::make_index_sequence<sizeof...(Is)>{});
  }

  template <typename CapType, typename Op, size_t PrefixRank>
  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) block_op_get_flat_dim_item(
      const Op &op,
      index_t item,
      const cuda::std::array<index_t, PrefixRank> &prefix_indices)
  {
    constexpr int rank = Op::Rank();
    static_assert(PrefixRank <= static_cast<size_t>(rank),
                  "Block CUB prefix rank cannot exceed operand rank");

    cuda::std::array<index_t, rank> scalar_indices{};
    for (int r = 0; r < static_cast<int>(PrefixRank); r++) {
      scalar_indices[r] = prefix_indices[r];
    }

    for (int r = rank - 1; r >= static_cast<int>(PrefixRank); r--) {
      const index_t dim_size = op.Size(r);
      scalar_indices[r] = item % dim_size;
      item /= dim_size;
    }

    auto value = block_op_get_value_at<CapType>(op,
                                                scalar_indices,
                                                cuda::std::make_index_sequence<rank>{});
    return GetVectorVal(value, 0);
  }

#ifndef __CUDACC_RTC__
  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ int GetCubBlockShmRequired(CubBlockAlgorithm algorithm,
                                                           ElementsPerThread ept,
                                                           int block_size)
  {
#if defined(MATX_EN_JIT) && !defined(__CUDACC_RTC__)
    const char *algorithm_name = "reduce";
    if (algorithm == CubBlockAlgorithm::SCAN) {
      algorithm_name = "scan";
    }
    else if (algorithm == CubBlockAlgorithm::SORT) {
      algorithm_name = "sort";
    }
    else if (algorithm == CubBlockAlgorithm::SORT_PAIRS) {
      algorithm_name = "sort_pairs";
    }

    return nvrtc_get_cub_block_shmem_size(algorithm_name,
                                          detail::type_to_string<T>(),
                                          static_cast<int>(ept),
                                          block_size);
#else
    (void)algorithm;
    (void)ept;
    (void)block_size;
    MATX_THROW(matxNotSupported, "CUB block shared-memory queries require MATX_EN_JIT");
    return 0;
#endif
  }
#endif

  template <typename _Tp = void>
  struct ProdReduce
  {
    [[nodiscard]] constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
      noexcept(noexcept(__lhs * __rhs))
    {
      return __lhs * __rhs;
    }
  };

  template <typename _Tp = void>
  struct MinReduce
  {
    [[nodiscard]] constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
      noexcept(noexcept(__lhs < __rhs))
    {
      return __rhs < __lhs ? __rhs : __lhs;
    }
  };

  template <typename _Tp = void>
  struct MaxReduce
  {
    [[nodiscard]] constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
      noexcept(noexcept(__lhs < __rhs))
    {
      return __lhs < __rhs ? __rhs : __lhs;
    }
  };

  template <typename T, BlockReduceType ReduceType>
  __MATX_INLINE__ __MATX_DEVICE__ constexpr T cub_identity_value()
  {
    if constexpr (ReduceType == BlockReduceType::SUM) {
      return T{0};
    }
    else if constexpr (ReduceType == BlockReduceType::MIN) {
      return cuda::std::numeric_limits<T>::max();
    }
    else if constexpr (ReduceType == BlockReduceType::MAX) {
      return cuda::std::numeric_limits<T>::lowest();
    }
    else {
      return T{1};
    }
  }

  template <typename T, BlockReduceType ReduceType>
  __MATX_INLINE__ __MATX_DEVICE__ T cub_combine_value(const T &lhs, const T &rhs)
  {
    if constexpr (ReduceType == BlockReduceType::SUM) {
      return lhs + rhs;
    }
    else if constexpr (ReduceType == BlockReduceType::MIN) {
      return rhs < lhs ? rhs : lhs;
    }
    else if constexpr (ReduceType == BlockReduceType::MAX) {
      return lhs < rhs ? rhs : lhs;
    }
    else {
      return lhs * rhs;
    }
  }

  template <typename CapType, BlockSortDirection Direction>
  struct BlockSort {
    template <typename sort_type>
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ int GetShmRequired() {
      using BlockRadixSort = cub::BlockRadixSort<sort_type, CapType::block_size, static_cast<int>(CapType::ept)>;
      return static_cast<int>(sizeof(typename BlockRadixSort::TempStorage));
    }

    template <typename Op, typename... Is>
    static __MATX_INLINE__ __MATX_DEVICE__ auto Run(const Op &op, Is... indices) {
      static constexpr int ept = static_cast<int>(CapType::ept);
      using sort_type = typename Op::value_type;
      using BlockRadixSort = cub::BlockRadixSort<sort_type, CapType::block_size, ept>;

      __shared__ typename BlockRadixSort::TempStorage temp_storage;

      using ret_type = cuda::std::conditional_t<CapType::ept == ElementsPerThread::ONE, sort_type, Vector<sort_type, ept>>;
      ret_type thread_data{};
      sort_type (&items)[ept] = reinterpret_cast<sort_type (&)[ept]>(thread_data);
      auto input_items = block_op_get_last_dim_vector<CapType>(op, indices...);
      #pragma unroll
      for (int i = 0; i < ept; i++) {
        items[i] = GetVectorVal(input_items, i);
      }

      if constexpr (Direction == BlockSortDirection::ASCENDING) {
        BlockRadixSort(temp_storage).Sort(items);
      }
      else {
        BlockRadixSort(temp_storage).SortDescending(items);
      }

      return thread_data;
    }
  };

  template <typename CapType, BlockSortDirection Direction>
  struct BlockArgsort {
    template <typename sort_type>
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ int GetShmRequired() {
      using BlockRadixSort = cub::BlockRadixSort<sort_type, CapType::block_size, static_cast<int>(CapType::ept), index_t>;
      return static_cast<int>(sizeof(typename BlockRadixSort::TempStorage));
    }

    template <typename Op, typename... Is>
    static __MATX_INLINE__ __MATX_DEVICE__ auto Run(const Op &op, Is... indices) {
      static constexpr int ept = static_cast<int>(CapType::ept);
      using sort_type = typename Op::value_type;
      using BlockRadixSort = cub::BlockRadixSort<sort_type, CapType::block_size, ept, index_t>;

      __shared__ typename BlockRadixSort::TempStorage temp_storage;

      using key_ret_type = cuda::std::conditional_t<CapType::ept == ElementsPerThread::ONE, sort_type, Vector<sort_type, ept>>;
      using value_ret_type = cuda::std::conditional_t<CapType::ept == ElementsPerThread::ONE, index_t, Vector<index_t, ept>>;
      key_ret_type thread_keys{};
      value_ret_type thread_values{};
      sort_type (&keys)[ept] = reinterpret_cast<sort_type (&)[ept]>(thread_keys);
      index_t (&values)[ept] = reinterpret_cast<index_t (&)[ept]>(thread_values);

      const index_t linear_base = static_cast<index_t>(threadIdx.x) * ept;
      auto input_items = block_op_get_last_dim_vector<CapType>(op, indices...);
      #pragma unroll
      for (int i = 0; i < ept; i++) {
        keys[i] = GetVectorVal(input_items, i);
        values[i] = linear_base + i;
      }

      if constexpr (Direction == BlockSortDirection::ASCENDING) {
        BlockRadixSort(temp_storage).Sort(keys, values);
      }
      else {
        BlockRadixSort(temp_storage).SortDescending(keys, values);
      }

      return thread_values;
    }
  };

  template <typename CapType>
  struct BlockScan {
    template <typename scan_type>
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ int GetShmRequired() {
      using BlockScanT = cub::BlockScan<scan_type, CapType::block_size>;
      return static_cast<int>(sizeof(typename BlockScanT::TempStorage));
    }

    template <typename Op, typename... Is>
    static __MATX_INLINE__ __MATX_DEVICE__ auto Run(const Op &op, Is... indices) {
      static constexpr int ept = static_cast<int>(CapType::ept);
      using scan_type = typename Op::value_type;
      using BlockScanT = cub::BlockScan<scan_type, CapType::block_size>;

      // CCCL BlockScan's TempStorage is a property of the block dimensions and
      // algorithm. ITEMS_PER_THREAD is supplied to the InclusiveSum overload
      // below rather than to the BlockScan class template.
      __shared__ typename BlockScanT::TempStorage temp_storage;

      using ret_type = cuda::std::conditional_t<CapType::ept == ElementsPerThread::ONE, scan_type, Vector<scan_type, ept>>;
      ret_type thread_data{};
      scan_type (&items)[ept] = reinterpret_cast<scan_type (&)[ept]>(thread_data);
      auto input_items = block_op_get_last_dim_vector<CapType>(op, indices...);
      #pragma unroll
      for (int i = 0; i < ept; i++) {
        items[i] = GetVectorVal(input_items, i);
      }

      // CUB BlockScan takes ITEMS_PER_THREAD on the InclusiveSum overload, not
      // as a BlockScan class template parameter.
      BlockScanT(temp_storage).template InclusiveSum<ept>(items, items);
      return thread_data;
    }
  };

  template <typename CapType, BlockReduceType ReduceType, int ReduceSize, bool ScalarLoads = false>
  struct BlockReduce {
    template <typename reduce_type>
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ int GetShmRequired() {
      using BlockReduceT = cub::BlockReduce<reduce_type, CapType::block_size>;
      return static_cast<int>(sizeof(typename BlockReduceT::TempStorage));
    }

    template <typename Op, typename... Is>
    static __MATX_INLINE__ __MATX_DEVICE__ auto RunLastDim(const Op &op, Is... indices) {
      static constexpr int ept = static_cast<int>(CapType::ept);
      using reduce_type = typename Op::value_type;
      using BlockReduceT = cub::BlockReduce<reduce_type, CapType::block_size>;

      __shared__ typename BlockReduceT::TempStorage temp_storage;
      cuda::std::array<index_t, sizeof...(Is)> prefix_indices{static_cast<index_t>(indices)...};

      reduce_type partial = cub_identity_value<reduce_type, ReduceType>();
      const int linear_base = static_cast<int>(threadIdx.x) * ept;
      static constexpr bool reduce_last_dim_only = (sizeof...(Is) + 1) == Op::Rank();

      if constexpr (CapType::pass_through_threads || ScalarLoads || !reduce_last_dim_only) {
        using ScalarCap = typename CapType::scalar_cap;
        #pragma unroll
        for (int i = 0; i < ept; i++) {
          const bool valid_item = linear_base + i < ReduceSize;
          if (valid_item || CapType::pass_through_threads) {
            const index_t item = valid_item ? static_cast<index_t>(linear_base + i) : static_cast<index_t>(ReduceSize - 1);
            auto value = block_op_get_flat_dim_item<ScalarCap>(op, item, prefix_indices);
            if (!valid_item) {
              continue;
            }
            partial = cub_combine_value<reduce_type, ReduceType>(
                partial,
                value);
          }
        }
      }
      else if constexpr (ept == 1) {
        if (linear_base < ReduceSize) {
          partial = op.template operator()<CapType>(indices..., static_cast<index_t>(threadIdx.x));
        }
      }
      else if constexpr ((ReduceSize % ept) == 0) {
        auto thread_data = op.template operator()<CapType>(indices..., static_cast<index_t>(threadIdx.x));
        reduce_type (&items)[ept] = reinterpret_cast<reduce_type (&)[ept]>(thread_data);
        #pragma unroll
        for (int i = 0; i < ept; i++) {
          partial = cub_combine_value<reduce_type, ReduceType>(partial, items[i]);
        }
      }
      else {
        using ScalarCap = typename CapType::scalar_cap;
        #pragma unroll
        for (int i = 0; i < ept; i++) {
          if (linear_base + i < ReduceSize) {
            partial = cub_combine_value<reduce_type, ReduceType>(
                partial,
                op.template operator()<ScalarCap>(indices..., static_cast<index_t>(linear_base + i)));
          }
        }
      }

      const int valid_threads = (ReduceSize + ept - 1) / ept;
      if constexpr (ReduceType == BlockReduceType::SUM) {
        return BlockReduceT(temp_storage).Sum(partial, valid_threads);
      }
      else if constexpr (ReduceType == BlockReduceType::MIN) {
        return BlockReduceT(temp_storage).Reduce(partial, MinReduce<reduce_type>{}, valid_threads);
      }
      else if constexpr (ReduceType == BlockReduceType::MAX) {
        return BlockReduceT(temp_storage).Reduce(partial, MaxReduce<reduce_type>{}, valid_threads);
      }
      else {
        return BlockReduceT(temp_storage).Reduce(partial, ProdReduce<reduce_type>{}, valid_threads);
      }
    }
  };

}
}

#endif
