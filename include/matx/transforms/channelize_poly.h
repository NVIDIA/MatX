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

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <numeric>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/kernels/channelize_poly.cuh"
#include "matx/operators/fft.h"
#include "matx/operators/slice.h"
#include <cuda/std/__algorithm/max.h>

namespace matx {
namespace detail {

// Any channel count at or below MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD will
// use the fused-channel kernel. If this is increased, then the switch statement
// in the fused kernel wrapper below must be adjusted to include the additional
// channel counts.
constexpr index_t MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD = 6;

// Number of output samples per channel per iteration for the kernel that stores
// the input data in shared memory. Ideally, this value would be determined dynamically
// to balance occupancy and CTA size. For now, we choose a reasonable default.
constexpr index_t MATX_CHANNELIZE_POLY1D_FULL_SMEM_KERNEL_NOUT_PER_ITER = 4;

// Maximum dynamic shared memory (bytes) for caching filter taps in ChannelizePoly1D.
constexpr size_t MATX_CHANNELIZE_POLY1D_MAX_FILTER_SMEM_BYTES = 6 * 1024;

// Constants for the SmemTiled kernel.
constexpr int MATX_CHANNELIZE_POLY1D_SMEM_TILED_CTILE = 64;
constexpr int MATX_CHANNELIZE_POLY1D_SMEM_TILED_NOUT  = 4;
constexpr size_t MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_BYTES = 48 * 1024;
// Maximum filter smem budget. When the filter exceeds this, it is read from
// global/L2 instead, freeing smem for better occupancy.
constexpr size_t MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_FILTER_BYTES = 2048;

// Compute the shared memory footprint for the SmemTiled filter taps.
// For D==M: one phase per tile channel → CTILE * P elements.
// For D<M: K phases per tile channel → CTILE * K * P elements.
// Returns a value exceeding the filter budget when K exceeds the rotation
// limit, which causes FilterInSmem=false in the dispatch.
template <typename FilterType>
inline size_t matxChannelizePoly1DInternal_SmemTiledFilterSmemBytes(
    index_t num_channels, index_t filter_len, index_t decimation_factor)
{
  using filter_t = typename FilterType::value_type;
  constexpr int CTILE = MATX_CHANNELIZE_POLY1D_SMEM_TILED_CTILE;
  const index_t P = (filter_len + num_channels - 1) / num_channels;
  if (decimation_factor == num_channels) {
    return static_cast<size_t>(CTILE) * P * sizeof(filter_t);
  }
  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const index_t K = num_channels / gcd_val;
  if (K > MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_ROTATIONS) {
    return MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_FILTER_BYTES + 1; // exceeds budget, so FilterInSmem=false
  }
  return static_cast<size_t>(CTILE) * K * P * sizeof(filter_t);
}

template <typename OutType, typename InType, typename FilterType>
inline size_t matxChannelizePoly1DInternal_SmemTiledSizeBytes(
    const OutType &o, const InType &, const FilterType &filter, index_t decimation_factor)
{
  using input_t  = typename InType::value_type;

  constexpr int CTILE = MATX_CHANNELIZE_POLY1D_SMEM_TILED_CTILE;
  constexpr int NOUT  = MATX_CHANNELIZE_POLY1D_SMEM_TILED_NOUT;

  const index_t M = o.Size(OutType::Rank() - 1);
  const index_t filter_len = filter.Size(FilterType::Rank() - 1);
  const index_t P = (filter_len + M - 1) / M;
  const size_t input_smem = static_cast<size_t>(P + NOUT - 1) * CTILE * sizeof(input_t);

  const size_t filter_smem = matxChannelizePoly1DInternal_SmemTiledFilterSmemBytes<FilterType>(M, filter_len, decimation_factor);
  if (filter_smem > MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_FILTER_BYTES) {
    return input_smem;
  }

  const size_t filter_smem_aligned = filter_smem +
      ((filter_smem % sizeof(input_t)) ? (sizeof(input_t) - filter_smem % sizeof(input_t)) : 0);
  return filter_smem_aligned + input_smem;
}

template <typename OutType, typename InType, typename FilterType>
inline bool matxChannelizePoly1DInternal_ShouldUseSmemTiledKernel(
    const OutType &o, const InType &in, const FilterType &filter, index_t decimation_factor)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  // Skip SmemTiled for small channel counts where most of CTILE would be
  // idle threads. The generic ChannelizePoly1D kernel is more efficient
  // in this regime.
  if (num_channels <= MATX_CHANNELIZE_POLY1D_SMEM_TILED_CTILE * 3 / 4) {
    return false;
  }
  // The input circular buffer must fit in smem. The filter may or may not
  // be included (FilterInSmem is decided separately).
  return matxChannelizePoly1DInternal_SmemTiledSizeBytes(o, in, filter, decimation_factor)
      <= MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_BYTES;
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void matxChannelizePoly1DInternal_SmemTiled(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  constexpr int CTILE = MATX_CHANNELIZE_POLY1D_SMEM_TILED_CTILE;
  constexpr int NOUT  = MATX_CHANNELIZE_POLY1D_SMEM_TILED_NOUT;

  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const index_t nout_per_channel = o.Size(OutType::Rank() - 2);
  const int num_batches = static_cast<int>(TotalSize(i) / i.Size(i.Rank() - 1));

  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const int32_t K = static_cast<int32_t>(num_channels / gcd_val);

  const int channel_tiles = static_cast<int>((num_channels + CTILE - 1) / CTILE);
  // Target ~1024 spatial blocks (time × channel tiles) to saturate the GPU.
  // For large channel counts, fewer time blocks are needed.
  const int target_time_blocks = cuda::std::max(1, (1024 + channel_tiles - 1) / channel_tiles);
  const int elem_per_block = static_cast<int>(
      (nout_per_channel + target_time_blocks - 1) / target_time_blocks);
  const int time_blocks = static_cast<int>(
      (nout_per_channel + elem_per_block - 1) / elem_per_block);

  dim3 block(CTILE, NOUT);
  dim3 grid(time_blocks, channel_tiles, num_batches);

  const index_t filter_len = filter.Size(FilterType::Rank() - 1);
  const bool filter_in_smem = matxChannelizePoly1DInternal_SmemTiledFilterSmemBytes<FilterType>(
      num_channels, filter_len, decimation_factor) <= MATX_CHANNELIZE_POLY1D_SMEM_TILED_MAX_FILTER_BYTES;
  const size_t smem_size = matxChannelizePoly1DInternal_SmemTiledSizeBytes(o, i, filter, decimation_factor);

  // Use int32_t for intra-kernel index arithmetic when all tensor dimensions
  // fit, avoiding 64-bit IMAD.WIDE instructions in the inner loops.
  const index_t input_len = i.Size(i.Rank() - 1);
  const bool use_32bit = (sizeof(index_t) <= sizeof(int32_t)) ||
      (static_cast<int64_t>(input_len) + num_channels <= std::numeric_limits<int32_t>::max() &&
       nout_per_channel <= std::numeric_limits<int32_t>::max() &&
       num_channels <= std::numeric_limits<int32_t>::max());

  // Dispatch on MaximallyDecimated x FilterInSmem x IndexType
  constexpr bool kMaxDec  = true;
  constexpr bool kOversampled = false;
  constexpr bool kFiltSmem = true;
  constexpr bool kFiltGlobal = false;

  auto launch = [&](auto idx_tag) {
    using IdxT = decltype(idx_tag);
    const IdxT epb = static_cast<IdxT>(elem_per_block);
    const IdxT df  = static_cast<IdxT>(decimation_factor);
    if (decimation_factor == num_channels) {
      if (filter_in_smem) {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kMaxDec, kFiltSmem, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      } else {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kMaxDec, kFiltGlobal, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      }
    } else {
      if (filter_in_smem) {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kOversampled, kFiltSmem, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      } else {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kOversampled, kFiltGlobal, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      }
    }
  };

  if constexpr (sizeof(index_t) <= sizeof(int32_t)) {
    launch(index_t{});
  } else {
    if (use_32bit) {
      launch(int32_t{});
    } else {
      launch(index_t{});
    }
  }
#endif
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void matxChannelizePoly1DInternal(OutType o, const InType &i,
                                     const FilterType &filter, index_t decimation_factor, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  using filter_t = typename FilterType::value_type;
  const index_t filter_len = filter.Size(FilterType::Rank()-1);

  const int THREADS = 256;
  const index_t ELTS_PER_THREAD = CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
  const int elem_blocks = static_cast<int>(
    (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD);
  dim3 grid(elem_blocks, static_cast<int>(num_channels), num_batches);
  if (decimation_factor == num_channels) {
    // For M == D, cache one filter phase in dynamic shared memory if it fits.
    const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;
    const size_t smem_needed = static_cast<size_t>(filter_phase_len) * sizeof(filter_t);
    const uint32_t smem_bytes = (smem_needed <= MATX_CHANNELIZE_POLY1D_MAX_FILTER_SMEM_BYTES)
        ? static_cast<uint32_t>(smem_needed) : 0;
    ChannelizePoly1D<THREADS, true, OutType, InType, FilterType, AccumType><<<grid, THREADS, smem_bytes, stream>>>(
        o, i, filter, decimation_factor, smem_bytes);
  } else {
    ChannelizePoly1D<THREADS, false, OutType, InType, FilterType, AccumType><<<grid, THREADS, 0, stream>>>(
        o, i, filter, decimation_factor, 0);
  }
#endif
}

template <typename OutType, typename InType, typename FilterType>
inline size_t matxChannelizePoly1DInternal_SmemSizeBytes(const OutType &o, const InType &, const FilterType &filter)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;

  index_t filter_len = filter.Size(FilterType::Rank()-1);

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;

  size_t smem_size = sizeof(filter_t)*(num_channels)*(filter_phase_len) +
    sizeof(input_t)*(num_channels)*(filter_phase_len + MATX_CHANNELIZE_POLY1D_FULL_SMEM_KERNEL_NOUT_PER_ITER - 1);
  const size_t max_sizeof = cuda::std::max(sizeof(filter_t), sizeof(input_t));
  if (smem_size % max_sizeof) {
    smem_size += max_sizeof - (smem_size % max_sizeof);
  }
  return smem_size;
}

template <typename OutType, typename InType, typename FilterType>
inline size_t matxChannelizePoly1DInternal_ShouldUseSmemKernel(const OutType &out, const InType &in, const FilterType &filter)
{
  // 48 KB is the largest shared memory allocation that does not require
  // explicit opt-in via cudaFuncSetAttribute()
  const size_t MAX_SMEM_BYTES = 48 * 1024;
  // The full shared memory kernel uses blocks of size
  // (num_channels, detail::MATX_CHANNELIZE_POLY1D_FULL_SMEM_KERNEL_NOUT_PER_ITER), so ensure
  // that the resulting thread per block count will not exceed MAX_NUM_THREADS_PER_BLOCK
  const int MAX_NUM_THREADS_PER_BLOCK = 1024;
  const index_t num_channels = out.Size(OutType::Rank()-1);
  return (
      matxChannelizePoly1DInternal_SmemSizeBytes(out, in, filter) <= MAX_SMEM_BYTES &&
      num_channels <= (MAX_NUM_THREADS_PER_BLOCK/detail::MATX_CHANNELIZE_POLY1D_FULL_SMEM_KERNEL_NOUT_PER_ITER));
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void matxChannelizePoly1DInternal_Smem(OutType o, const InType &i, const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  const int target_num_blocks = 1024;
  const int elem_per_block = static_cast<int>(
    (nout_per_channel + target_num_blocks - 1) / target_num_blocks);
  dim3 block(static_cast<int>(num_channels), MATX_CHANNELIZE_POLY1D_FULL_SMEM_KERNEL_NOUT_PER_ITER);
  const uint32_t num_blocks = static_cast<uint32_t>((nout_per_channel + elem_per_block - 1) / elem_per_block);
  dim3 grid(num_blocks, 1, num_batches);
  const size_t smem_size = matxChannelizePoly1DInternal_SmemSizeBytes(o, i, filter);
  ChannelizePoly1D_Smem<OutType, InType, FilterType, AccumType><<<grid, block, smem_size, stream>>>(
      o, i, filter, elem_per_block);
#endif
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void matxChannelizePoly1DInternal_FusedChan(OutType o, const InType &i,
                                     const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  const int THREADS = 256;
  const index_t ELTS_PER_THREAD = CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
  const int elem_blocks = static_cast<int>(
    (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD);
  dim3 grid(elem_blocks, 1, num_batches);
  switch (num_channels) {
    case 2:
      ChannelizePoly1D_FusedChan<THREADS, 2, OutType, InType, FilterType, AccumType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 3:
      ChannelizePoly1D_FusedChan<THREADS, 3, OutType, InType, FilterType, AccumType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 4:
      ChannelizePoly1D_FusedChan<THREADS, 4, OutType, InType, FilterType, AccumType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 5:
      ChannelizePoly1D_FusedChan<THREADS, 5, OutType, InType, FilterType, AccumType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 6:
      ChannelizePoly1D_FusedChan<THREADS, 6, OutType, InType, FilterType, AccumType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    default:
      MATX_THROW(matxInvalidDim, "channelize_poly: channel count not support with fused kernel");
  }
#endif
}

template <typename DataType>
inline void matxChannelizePoly1DUnpackInternal(DataType inout, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  constexpr int THREADS = 128;
  const index_t num_elem_per_channel = inout.Size(DataType::Rank()-2);
  const index_t num_channels = inout.Size(DataType::Rank()-1);
  const int num_batches = static_cast<int>(TotalSize(inout)/
    (num_channels * num_elem_per_channel));
  const int gy = static_cast<int>((num_elem_per_channel + THREADS - 1) / THREADS);
  const dim3 grid(num_batches, gy);
  ChannelizePoly1DUnpackDFT<<<grid, THREADS, 0, stream>>>(inout);
#endif
}

} // end namespace detail

/**
 * @brief 1D polyphase channelizer. A channelizer separates an input signal into a set of
 * constituent channels, each corresponding to a band of the input signal bandwidth. Supports both
 * maximally decimated (critically sampled, decimation_factor == num_channels) and oversampled
 * (decimation_factor < num_channels) cases, including rational oversampling ratios.
 * 
 * @tparam OutType Type of output
 * @tparam InType Type of input
 * @tparam FilterType Type of filter
 * @tparam AccumType Type of accumulator. This type should always be real, but it will be promoted to
 * complex when necessary.
 * @param out Output tensor
 * @param in Input operator
 * @param f Filter operator
 * @param num_channels Number of channels in which to separate the signal. Must be greater than 1.
 * @param decimation_factor Factor by which to downsample the input signal into the channels. When
 * decimation_factor equals num_channels, this is the maximally decimated (critically sampled) case.
 * When decimation_factor is less than num_channels, this is the oversampled case with overlapping
 * channels. Both integer (num_channels % decimation_factor == 0) and rational oversampling ratios
 * are supported.
 * @param stream CUDA stream on which to run the kernel(s)
 */
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void channelize_poly_impl(OutType out, const InType &in, const FilterType &f,
                   index_t num_channels, index_t decimation_factor, cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using OutputOp = std::remove_cv_t<std::remove_reference_t<OutType>>;
  using InputOp = std::remove_cv_t<std::remove_reference_t<InType>>;
  using FilterOp = std::remove_cv_t<std::remove_reference_t<FilterType>>;
  using input_t = typename InputOp::value_type;
  using filter_t = typename FilterOp::value_type;
  using output_t = typename OutputOp::value_type;

  constexpr int IN_RANK = InputOp::Rank();
  constexpr int OUT_RANK = OutputOp::Rank();

  // The last dimension of the input becomes [num_channels, num_elem_per_channel] in the last
  // two dimensions of the output
  MATX_STATIC_ASSERT_STR(OUT_RANK == IN_RANK+1, matxInvalidDim, "channelize_poly: output rank should be 1 higher than input");

  MATX_STATIC_ASSERT_STR(is_complex_v<output_t> || is_complex_half_v<output_t>,
    matxInvalidType, "channelize_poly: output type must be complex");

  // Currently only support 1D filters.
  MATX_STATIC_ASSERT_STR(FilterType::Rank() == 1, matxInvalidDim, "channelize_poly: currently only support 1D filters");

  MATX_ASSERT_STR(num_channels > 1, matxInvalidParameter,
    "channelize_poly: num_channels must be greater than 1");
  MATX_ASSERT_STR(decimation_factor > 0, matxInvalidParameter,
    "channelize_poly: decimation_factor must be positive");
  MATX_ASSERT_STR(decimation_factor <= num_channels, matxInvalidParameter,
    "channelize_poly: decimation_factor must be <= num_channels");

  for(int i = 0 ; i < IN_RANK-1; i++) {
    MATX_ASSERT_STR(out.Size(i) == in.Size(i), matxInvalidDim, "channelize_poly: input/output must have matched batch sizes");
  }

  [[maybe_unused]] const index_t num_elem_per_channel = (in.Size(IN_RANK-1) + decimation_factor - 1) / decimation_factor;

  MATX_ASSERT_STR(out.Size(OUT_RANK-1) == num_channels, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-1 mismatch");
  MATX_ASSERT_STR(out.Size(OUT_RANK-2) == num_elem_per_channel, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-2 mismatch");

  // If neither the input nor the filter is complex, then the filtered samples will be real-valued
  // and we will use an R2C transform. Otherwise, we will use a C2C transform.
  if constexpr (! is_complex_v<input_t> && ! is_complex_half_v<input_t> && ! is_complex_v<filter_t> && ! is_complex_half_v<filter_t>) {
    // The fused-DFT kernel only supports the maximally decimated case (D == M).
    if (decimation_factor == num_channels && num_channels <= detail::MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD) {
      matxChannelizePoly1DInternal_FusedChan<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
    } else {
      index_t start_dims[OUT_RANK], stop_dims[OUT_RANK];
      std::fill_n(start_dims, OUT_RANK, 0);
      std::fill_n(stop_dims, OUT_RANK, matxEnd);

      // The first kernel below needs a buffer of type input_t (known to be real in this
      // constexpr branch) into which we store filtered data prior to the real-to-complex
      // FFT. If the output buffer is contiguous, then we use an aliased tensor view of type input_t
      // for that buffer where the last dimension is twice as large (because input_t is real
      // and output_t is complex). We then use a slice to maintain the expected dimensions.
      // If the output buffer is not contiguous, then we async allocate a temporary buffer.
      // There is one caveat with this allocate: the batched fft implementation currently
      // requires that all input pointers must be aligned to the corresponding complex type,
      // which cannot be guaranteed to always be true for a real-valued tensor. This was
      // not an issue for the reused output buffer because the output tensor is complex-valued,
      // so we always have an even stride from one batch to the next. As a temporary workaround
      // for the FFT alignment issue, we add one channel in the odd-channel case and use a
      // slice to create a tensor view of only [0, num_channels-1]. This guarantees that we
      // always stride by an even number of elements from one batch to the next while exposing
      // a tensor view of appropriate dimensions.
      using post_filter_t = typename inner_op_type_t<output_t>::type;
      auto fft_in_slice = [&out, &start_dims, &stop_dims, num_channels, stream]() -> auto {
        auto fft_in_shape = out.Shape();
        if (out.IsContiguous()) {
          fft_in_shape[OUT_RANK-1] *= 2;
          auto fft_in = make_tensor<post_filter_t>(reinterpret_cast<post_filter_t*>(out.Data()), fft_in_shape);
          stop_dims[OUT_RANK-1] = num_channels;
          return slice<OUT_RANK>(fft_in, start_dims, stop_dims);
        } else {
          if (num_channels % 2 == 1) {
            fft_in_shape[OUT_RANK-1]++;
            stop_dims[OUT_RANK-1] = num_channels;
          }
          auto tmp = make_tensor<post_filter_t>(fft_in_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
          return slice<OUT_RANK>(tmp, start_dims, stop_dims);
        }
      }();

      if (decimation_factor == num_channels && matxChannelizePoly1DInternal_ShouldUseSmemKernel(out, in, f)) {
        matxChannelizePoly1DInternal_Smem<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, stream);
      } else if (matxChannelizePoly1DInternal_ShouldUseSmemTiledKernel(out, in, f, decimation_factor)) {
        matxChannelizePoly1DInternal_SmemTiled<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, decimation_factor, stream);
      } else {
        matxChannelizePoly1DInternal<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, decimation_factor, stream);
      }
      stop_dims[OUT_RANK-1] = (num_channels/2) + 1;
      auto out_packed = slice<OUT_RANK>(out, start_dims, stop_dims);
      (out_packed = fft(fft_in_slice, num_channels)).run(stream);
      matxChannelizePoly1DUnpackInternal(out, stream);
    }
  } else {
    // The fused-DFT kernel only supports the maximally decimated case (D == M).
    if (decimation_factor == num_channels && num_channels <= detail::MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD) {
      matxChannelizePoly1DInternal_FusedChan<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
    } else {
      if (decimation_factor == num_channels && matxChannelizePoly1DInternal_ShouldUseSmemKernel(out, in, f)) {
        matxChannelizePoly1DInternal_Smem<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
      } else if (matxChannelizePoly1DInternal_ShouldUseSmemTiledKernel(out, in, f, decimation_factor)) {
        matxChannelizePoly1DInternal_SmemTiled<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, decimation_factor, stream);
      } else {
        matxChannelizePoly1DInternal<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, decimation_factor, stream);
      }
      // Specify FORWARD here to prevent any normalization after the ifft. We do not
      // want any extra scaling on the output values.
      (out = ifft(out, num_channels, FFTNorm::FORWARD)).run(stream);
    }
  }
}
} // end namespace matx
