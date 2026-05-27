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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <numeric>
#include <vector>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/executors/host.h"
#include "matx/kernels/channelize_poly.cuh"
#include "matx/operators/fft.h"
#include "matx/operators/slice.h"
#include <cuda/std/__algorithm/max.h>

namespace matx {
namespace detail {
namespace cpoly {

// Any channel count at or below FusedChanThreshold will
// use the fused-channel kernel. If this is increased, then the switch statement
// in the fused kernel wrapper below must be adjusted to include the additional
// channel counts.
constexpr index_t FusedChanThreshold = 6;

// Number of output samples per channel per iteration for the kernel that stores
// the input data in shared memory. Ideally, this value would be determined dynamically
// to balance occupancy and CTA size. For now, we choose a reasonable default.
constexpr index_t FullSmemKernelNoutPerIter = 4;

// Maximum dynamic shared memory (bytes) for caching filter taps in ChannelizePoly1D.
constexpr size_t GenericMaxFilterSmemBytes = 6 * 1024;

// Constants for the SmemTiled kernel. Two CTILE sizes are instantiated:
//   CTILE=32 for num_channels <= 32: single channel tile, 100% thread
//            utilization. Block size: 32 * NOUT = 128 threads.
//   CTILE=64 for num_channels > 32: original shape. Block size: 64 * NOUT
//            = 256 threads.
// NOUT=4 across both variants; NOUT=8 for the CTILE=32 path was evaluated
// and was ~5% slower (doubles block size without increasing elem_per_block,
// so fewer blocks/SM and no occupancy win).
constexpr int SmemTiledCtile = 64;
constexpr int SmemTiledCtileSmall = 32;
constexpr int SmemTiledNout  = 4;
constexpr size_t SmemTiledMaxBytes = 48 * 1024;
// Maximum filter smem budget. A filter is only considered for smem
// residency if its footprint (in whichever layout we're considering, Full
// or Rotated) fits under this cap. Raising this costs occupancy by
// inflating per-block smem, so keep it small.
constexpr size_t SmemTiledMaxFilterBytes = 4096;

// Filter-smem placement strategy chosen by the dispatcher and forwarded to
// the kernel via template parameters.
//   Full:    one copy of each tap at smem[p * M + phase]. Footprint M*P.
//            No per-channel or per-rotation duplication; inner loop does a
//            per-(c,k) phase compute.
//   Rotated: [channel][k][p] redundant layout, footprint CTILE*K*P (or
//            CTILE*P for D==M). Direct indexing, larger for oversampled
//            configs with small num_channels, smaller for large num_channels.
//   Global:  filter stays in GMEM, loaded through L1 in the inner loop.
enum class SmemTiledFilterLayout { Full, Rotated, Global };

// Pick the CTILE to use for this launch: 32 when num_channels fits in a
// single 32-wide tile, 64 otherwise. Centralized so dispatch, sizing, and
// launch agree.
constexpr int SmemTiledCtileFor(index_t num_channels)
{
  return (num_channels <= SmemTiledCtileSmall)
      ? SmemTiledCtileSmall
      : SmemTiledCtile;
}

// Compute the shared memory footprint for the SmemTiled filter taps in the
// Rotated layout (one [channel][k][p] block per CTILE, K phases duplicated
// across channels, zero-padded when CTILE > M).
// For D==M: one phase per tile channel -> CTILE * P elements.
// For D<M: K phases per tile channel -> CTILE * K * P elements.
// Returns a value exceeding the filter budget when K exceeds the rotation
// limit, which causes FilterInSmem=false in the dispatch.
template <typename FilterType>
inline size_t SmemTiledFilterBytesRotated(
    index_t num_channels, index_t filter_len, index_t decimation_factor, int ctile)
{
  using filter_t = typename FilterType::value_type;
  const index_t P = (filter_len + num_channels - 1) / num_channels;
  if (decimation_factor == num_channels) {
    return static_cast<size_t>(ctile) * P * sizeof(filter_t);
  }
  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const index_t K = num_channels / gcd_val;
  if (K > SmemTiledMaxRotations) {
    return SmemTiledMaxFilterBytes + 1; // exceeds budget, so FilterInSmem=false
  }
  return static_cast<size_t>(ctile) * K * P * sizeof(filter_t);
}

// Shared memory footprint for the Full filter layout: M * P unique taps,
// no per-channel or per-rotation duplication. Full's footprint scales with
// num_channels while Rotated's scales with CTILE*K (D<M) or CTILE (D==M),
// so which layout is smaller depends on the parameters: Full tends to be
// smaller for small num_channels, Rotated for large num_channels.
template <typename FilterType>
inline size_t SmemTiledFilterBytesFull(
    index_t num_channels, index_t filter_len)
{
  using filter_t = typename FilterType::value_type;
  const index_t P = (filter_len + num_channels - 1) / num_channels;
  return static_cast<size_t>(num_channels) * P * sizeof(filter_t);
}

// Pick a filter-smem layout for this dispatch. Of the candidates that fit
// under the filter budget (Full, Rotated), choose the one with the smaller
// footprint to maximize occupancy. Fall back to Global when neither fits.
// The caller must separately verify that filter + input both fit under
// MAX_BYTES.
template <typename OutType, typename FilterType>
inline SmemTiledFilterLayout SmemTiledChooseFilterLayout(
    const OutType &o, const FilterType &filter, index_t decimation_factor, int ctile)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const index_t filter_len = filter.Size(FilterType::Rank() - 1);
  const size_t full_bytes = SmemTiledFilterBytesFull<FilterType>(
      num_channels, filter_len);
  const size_t rotated_bytes = SmemTiledFilterBytesRotated<FilterType>(
      num_channels, filter_len, decimation_factor, ctile);
  const bool full_fits = full_bytes <= SmemTiledMaxFilterBytes;
  const bool rotated_fits = rotated_bytes <= SmemTiledMaxFilterBytes;
  if (full_fits && rotated_fits) {
    return (full_bytes <= rotated_bytes)
        ? SmemTiledFilterLayout::Full
        : SmemTiledFilterLayout::Rotated;
  }
  if (full_fits) {
    return SmemTiledFilterLayout::Full;
  }
  if (rotated_fits) {
    return SmemTiledFilterLayout::Rotated;
  }
  return SmemTiledFilterLayout::Global;
}

template <typename OutType, typename InType, typename FilterType>
inline size_t SmemTiledSizeBytes(
    const OutType &o, const InType &, const FilterType &filter, index_t decimation_factor, int ctile)
{
  using input_t  = typename InType::value_type;

  constexpr int NOUT = SmemTiledNout;

  const index_t M = o.Size(OutType::Rank() - 1);
  const index_t filter_len = filter.Size(FilterType::Rank() - 1);
  const index_t P = (filter_len + M - 1) / M;
  const size_t input_smem = static_cast<size_t>(P + NOUT - 1) * ctile * sizeof(input_t);

  const auto layout = SmemTiledChooseFilterLayout(
      o, filter, decimation_factor, ctile);
  if (layout == SmemTiledFilterLayout::Global) {
    return input_smem;
  }

  const size_t filter_smem = (layout == SmemTiledFilterLayout::Full)
      ? SmemTiledFilterBytesFull<FilterType>(M, filter_len)
      : SmemTiledFilterBytesRotated<FilterType>(M, filter_len, decimation_factor, ctile);

  const size_t filter_smem_aligned = filter_smem +
      ((filter_smem % sizeof(input_t)) ? (sizeof(input_t) - filter_smem % sizeof(input_t)) : 0);
  return filter_smem_aligned + input_smem;
}

template <typename OutType, typename InType, typename FilterType>
inline bool ShouldUseSmemTiled(
    const OutType &o, const InType &in, const FilterType &filter, index_t decimation_factor)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const int ctile = SmemTiledCtileFor(num_channels);
  // The input circular buffer must fit in smem. The filter may or may not
  // be included (FilterInSmem is decided separately).
  return SmemTiledSizeBytes(o, in, filter, decimation_factor, ctile)
      <= SmemTiledMaxBytes;
}

template <typename AccumT, typename FilterT>
__MATX_HOST__ __MATX_INLINE__ auto HostChannelizeCastFilter(FilterT v)
{
  if constexpr (is_complex_v<FilterT>) {
    return static_cast<AccumT>(v);
  } else if constexpr (is_complex_v<AccumT>) {
    using accum_scalar_t = typename inner_op_type_t<AccumT>::type;
    return static_cast<accum_scalar_t>(v);
  } else {
    return static_cast<AccumT>(v);
  }
}

template <typename AccumT, typename InputT>
__MATX_HOST__ __MATX_INLINE__ auto HostChannelizeCastInput(InputT v)
{
  if constexpr (is_complex_v<InputT>) {
    return static_cast<AccumT>(v);
  } else if constexpr (is_complex_v<AccumT>) {
    using accum_scalar_t = typename inner_op_type_t<AccumT>::type;
    return static_cast<accum_scalar_t>(v);
  } else {
    return static_cast<AccumT>(v);
  }
}

template <typename AccumT, typename FilterValT, typename InputValT>
__MATX_HOST__ __MATX_INLINE__ void HostChannelizeCmac(
    AccumT &accum, FilterValT hv, InputValT iv)
{
  if constexpr (is_complex_v<AccumT> && is_complex_v<FilterValT> && is_complex_v<InputValT>) {
    auto h_re = hv.real(), h_im = hv.imag();
    auto i_re = iv.real(), i_im = iv.imag();
    auto a_re = accum.real(), a_im = accum.imag();
    a_re = h_re * i_re + a_re;
    a_re = -(h_im * i_im) + a_re;
    a_im = h_re * i_im + a_im;
    a_im = h_im * i_re + a_im;
    accum = {a_re, a_im};
  } else if constexpr (is_complex_v<AccumT> && !is_complex_v<FilterValT> && is_complex_v<InputValT>) {
    auto a_re = accum.real(), a_im = accum.imag();
    a_re = hv * iv.real() + a_re;
    a_im = hv * iv.imag() + a_im;
    accum = {a_re, a_im};
  } else if constexpr (is_complex_v<AccumT> && is_complex_v<FilterValT> && !is_complex_v<InputValT>) {
    auto a_re = accum.real(), a_im = accum.imag();
    a_re = hv.real() * iv + a_re;
    a_im = hv.imag() * iv + a_im;
    accum = {a_re, a_im};
  } else {
    accum += hv * iv;
  }
}

template <typename Op, typename Arr, size_t... Is>
__MATX_HOST__ __MATX_INLINE__ decltype(auto) HostReadSignalImpl(
    const Op &op, const Arr &batch_idx, index_t sample_idx,
    cuda::std::index_sequence<Is...>)
{
  return op(batch_idx[Is]..., sample_idx);
}

template <typename Op, typename Arr>
__MATX_HOST__ __MATX_INLINE__ decltype(auto) HostReadSignal(
    const Op &op, const Arr &batch_idx, index_t sample_idx)
{
  return HostReadSignalImpl(
      op, batch_idx, sample_idx,
      cuda::std::make_index_sequence<static_cast<size_t>(Op::Rank() - 1)>{});
}

template <typename OutType, typename Arr, typename ValueT, size_t... Is>
__MATX_HOST__ __MATX_INLINE__ void HostWriteOutputImpl(
    OutType &out, const Arr &batch_idx, index_t output_idx, index_t channel,
    const ValueT &value, cuda::std::index_sequence<Is...>)
{
  out(batch_idx[Is]..., output_idx, channel) =
      static_cast<typename OutType::value_type>(value);
}

template <typename OutType, typename Arr, typename ValueT>
__MATX_HOST__ __MATX_INLINE__ void HostWriteOutput(
    OutType &out, const Arr &batch_idx, index_t output_idx, index_t channel,
    const ValueT &value)
{
  HostWriteOutputImpl(
      out, batch_idx, output_idx, channel, value,
      cuda::std::make_index_sequence<static_cast<size_t>(OutType::Rank() - 2)>{});
}

template <typename ComplexAccumT, typename ValueT>
__MATX_HOST__ __MATX_INLINE__ ComplexAccumT HostAsComplex(ValueT v)
{
  using scalar_t = typename inner_op_type_t<ComplexAccumT>::type;
  if constexpr (is_complex_v<ValueT>) {
    return static_cast<ComplexAccumT>(v);
  } else {
    return ComplexAccumT{static_cast<scalar_t>(v), static_cast<scalar_t>(0)};
  }
}

template <typename ComplexAccumT>
__MATX_HOST__ __MATX_INLINE__ ComplexAccumT HostTwiddle(index_t channel, index_t branch, index_t num_channels)
{
  using scalar_t = typename inner_op_type_t<ComplexAccumT>::type;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double arg = 2.0 * pi * static_cast<double>(channel) *
      static_cast<double>(branch) / static_cast<double>(num_channels);
  return ComplexAccumT{
      static_cast<scalar_t>(std::cos(arg)),
      static_cast<scalar_t>(std::sin(arg))};
}

template <int CTILE, typename OutType, typename InType, typename FilterType, typename AccumType>
inline void SmemTiledImpl(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  constexpr int NOUT = SmemTiledNout;

  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const index_t nout_per_channel = o.Size(OutType::Rank() - 2);
  const int num_batches = static_cast<int>(TotalSize(i) / i.Size(i.Rank() - 1));

  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const int32_t K = static_cast<int32_t>(num_channels / gcd_val);

  const int channel_tiles = static_cast<int>((num_channels + CTILE - 1) / CTILE);
  // Target ~1024 spatial blocks (time * channel tiles) to saturate the GPU.
  // For large channel counts, fewer time blocks are needed.
  const int target_time_blocks = cuda::std::max(1, (1024 + channel_tiles - 1) / channel_tiles);
  const int elem_per_block = static_cast<int>(
      (nout_per_channel + target_time_blocks - 1) / target_time_blocks);
  const int time_blocks = static_cast<int>(
      (nout_per_channel + elem_per_block - 1) / elem_per_block);

  dim3 block(CTILE, NOUT);
  dim3 grid(time_blocks, channel_tiles, num_batches);

  const auto filter_layout = SmemTiledChooseFilterLayout(
      o, filter, decimation_factor, CTILE);
  const size_t smem_size = SmemTiledSizeBytes(o, i, filter, decimation_factor, CTILE);

  // Use int32_t for intra-kernel index arithmetic when all tensor dimensions
  // fit, avoiding 64-bit IMAD.WIDE instructions in the inner loops.
  const index_t input_len = i.Size(i.Rank() - 1);
  const bool use_32bit = (sizeof(index_t) <= sizeof(int32_t)) ||
      (static_cast<int64_t>(input_len) + num_channels <= std::numeric_limits<int32_t>::max() &&
       nout_per_channel <= std::numeric_limits<int32_t>::max() &&
       num_channels <= std::numeric_limits<int32_t>::max());

  // Dispatch on MaximallyDecimated x (FilterInSmem, FilterFullLayout) x
  // IndexType x IsUnitStride. Filter-smem layout selection: Full (smallest
  // footprint, phase compute at access), Rotated (direct indexing, redundant
  // storage), or Global (filter stays in GMEM).
  [[maybe_unused]] constexpr bool kMaxDec  = true;
  [[maybe_unused]] constexpr bool kOversampled = false;

  // Unit-stride fast path eligibility + runtime check: same pattern as
  // sar_bp / ChannelizePoly1D. Computed ops without .Data() / .Stride() fall
  // through to the slow-path (operator()) instantiation.
  constexpr bool fast_path_eligible =
      is_tensor_view_v<OutType> &&
      is_tensor_view_v<InType> &&
      is_tensor_view_v<FilterType>;

  auto launch = [&](auto idx_tag, auto is_unit_c) {
    using IdxT = decltype(idx_tag);
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    const IdxT epb = static_cast<IdxT>(elem_per_block);
    const IdxT df  = static_cast<IdxT>(decimation_factor);

    auto launch_with_layout = [&](auto in_smem_c, auto full_c) {
      constexpr bool FIS = decltype(in_smem_c)::value;
      constexpr bool FFL = decltype(full_c)::value;
      if (decimation_factor == num_channels) {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kMaxDec, FIS, FFL, IsUnitStride, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      } else {
        ChannelizePoly1D_SmemTiled<CTILE, NOUT, kOversampled, FIS, FFL, IsUnitStride, IdxT, OutType, InType, FilterType, AccumType>
            <<<grid, block, smem_size, stream>>>(o, i, filter, epb, df, K);
      }
    };

    switch (filter_layout) {
      case SmemTiledFilterLayout::Full:
        launch_with_layout(cuda::std::bool_constant<true>{}, cuda::std::bool_constant<true>{});
        break;
      case SmemTiledFilterLayout::Rotated:
        launch_with_layout(cuda::std::bool_constant<true>{}, cuda::std::bool_constant<false>{});
        break;
      case SmemTiledFilterLayout::Global:
        launch_with_layout(cuda::std::bool_constant<false>{}, cuda::std::bool_constant<false>{});
        break;
    }
  };

  auto dispatch = [&](auto idx_tag) {
    if constexpr (fast_path_eligible) {
      const bool is_unit_stride =
          o.Stride(OutType::Rank() - 1) == 1 &&
          i.Stride(InType::Rank() - 1) == 1 &&
          filter.Stride(FilterType::Rank() - 1) == 1;
      if (is_unit_stride) {
        launch(idx_tag, cuda::std::bool_constant<true>{});
      } else {
        launch(idx_tag, cuda::std::bool_constant<false>{});
      }
    } else {
      launch(idx_tag, cuda::std::bool_constant<false>{});
    }
  };

  if constexpr (sizeof(index_t) <= sizeof(int32_t)) {
    dispatch(index_t{});
  } else {
    if (use_32bit) {
      dispatch(int32_t{});
    } else {
      dispatch(index_t{});
    }
  }
#endif
}

// Wrapper that picks CTILE=32 (single-tile, full thread utilization) for
// num_channels <= 32 and CTILE=64 otherwise. Previously this path was
// rejected for num_channels <= 48 and fell through to the generic
// ChannelizePoly1D kernel, which does uncoalesced strided global loads and
// is ~3x slower.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void SmemTiled(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, cudaStream_t stream)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  if (num_channels <= SmemTiledCtileSmall) {
    SmemTiledImpl<
        SmemTiledCtileSmall,
        OutType, InType, FilterType, AccumType>(o, i, filter, decimation_factor, stream);
  } else {
    SmemTiledImpl<
        SmemTiledCtile,
        OutType, InType, FilterType, AccumType>(o, i, filter, decimation_factor, stream);
  }
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void Generic(OutType o, const InType &i,
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
  const index_t ELTS_PER_THREAD = ElemsPerThread * THREADS;
  const int elem_blocks = static_cast<int>(
    (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD);
  dim3 grid(elem_blocks, static_cast<int>(num_channels), num_batches);

  // Unit-stride fast path: only viable when every hot tensor is a
  // storage-backed view (Data()/Stride() callable) AND each one's last-dim
  // stride is 1. Runtime-check those strides and dispatch to the specialized
  // kernel via a bool_constant lambda.
  constexpr bool fast_path_eligible =
      is_tensor_view_v<OutType> &&
      is_tensor_view_v<InType> &&
      is_tensor_view_v<FilterType>;

  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    if (decimation_factor == num_channels) {
      // For M == D, cache one filter phase in dynamic shared memory if it fits.
      const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;
      const size_t smem_needed = static_cast<size_t>(filter_phase_len) * sizeof(filter_t);
      const uint32_t smem_bytes = (smem_needed <= GenericMaxFilterSmemBytes)
          ? static_cast<uint32_t>(smem_needed) : 0;
      ChannelizePoly1D<THREADS, true, IsUnitStride, OutType, InType, FilterType, AccumType>
          <<<grid, THREADS, smem_bytes, stream>>>(o, i, filter, decimation_factor, smem_bytes);
    } else {
      ChannelizePoly1D<THREADS, false, IsUnitStride, OutType, InType, FilterType, AccumType>
          <<<grid, THREADS, 0, stream>>>(o, i, filter, decimation_factor, 0);
    }
  };

  if constexpr (fast_path_eligible) {
    const bool is_unit_stride =
        o.Stride(OutType::Rank() - 1) == 1 &&
        i.Stride(InType::Rank() - 1) == 1 &&
        filter.Stride(FilterType::Rank() - 1) == 1;
    if (is_unit_stride) {
      launch(cuda::std::bool_constant<true>{});
    } else {
      launch(cuda::std::bool_constant<false>{});
    }
  } else {
    launch(cuda::std::bool_constant<false>{});
  }
#endif
}

template <typename OutType, typename InType, typename FilterType>
inline size_t SmemSizeBytes(const OutType &o, const InType &, const FilterType &filter)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;

  index_t filter_len = filter.Size(FilterType::Rank()-1);

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;

  size_t smem_size = sizeof(filter_t)*(num_channels)*(filter_phase_len) +
    sizeof(input_t)*(num_channels)*(filter_phase_len + FullSmemKernelNoutPerIter - 1);
  const size_t max_sizeof = cuda::std::max(sizeof(filter_t), sizeof(input_t));
  if (smem_size % max_sizeof) {
    smem_size += max_sizeof - (smem_size % max_sizeof);
  }
  return smem_size;
}

template <typename OutType, typename InType, typename FilterType>
inline size_t ShouldUseSmem(const OutType &out, const InType &in, const FilterType &filter)
{
  // 48 KB is the largest shared memory allocation that does not require
  // explicit opt-in via cudaFuncSetAttribute()
  const size_t MAX_SMEM_BYTES = 48 * 1024;
  // The full shared memory kernel uses blocks of size
  // (num_channels, FullSmemKernelNoutPerIter), so ensure
  // that the resulting thread per block count will not exceed MAX_NUM_THREADS_PER_BLOCK
  const int MAX_NUM_THREADS_PER_BLOCK = 1024;
  const index_t num_channels = out.Size(OutType::Rank()-1);
  return (
      SmemSizeBytes(out, in, filter) <= MAX_SMEM_BYTES &&
      num_channels <= (MAX_NUM_THREADS_PER_BLOCK/FullSmemKernelNoutPerIter));
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void Smem(OutType o, const InType &i, const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  const int target_num_blocks = 1024;
  const int elem_per_block = static_cast<int>(
    (nout_per_channel + target_num_blocks - 1) / target_num_blocks);
  dim3 block(static_cast<int>(num_channels), FullSmemKernelNoutPerIter);
  const uint32_t num_blocks = static_cast<uint32_t>((nout_per_channel + elem_per_block - 1) / elem_per_block);
  dim3 grid(num_blocks, 1, num_batches);
  const size_t smem_size = SmemSizeBytes(o, i, filter);

  constexpr bool fast_path_eligible =
      is_tensor_view_v<OutType> &&
      is_tensor_view_v<InType> &&
      is_tensor_view_v<FilterType>;
  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    ChannelizePoly1D_Smem<IsUnitStride, OutType, InType, FilterType, AccumType>
        <<<grid, block, smem_size, stream>>>(o, i, filter, elem_per_block);
  };
  if constexpr (fast_path_eligible) {
    const bool is_unit_stride =
        o.Stride(OutType::Rank() - 1) == 1 &&
        i.Stride(InType::Rank() - 1) == 1 &&
        filter.Stride(FilterType::Rank() - 1) == 1;
    if (is_unit_stride) {
      launch(cuda::std::bool_constant<true>{});
    } else {
      launch(cuda::std::bool_constant<false>{});
    }
  } else {
    launch(cuda::std::bool_constant<false>{});
  }
#endif
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void FusedChan(OutType o, const InType &i,
                                     const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  const int THREADS = 256;
  const index_t ELTS_PER_THREAD = ElemsPerThread * THREADS;
  const int elem_blocks = static_cast<int>(
    (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD);
  dim3 grid(elem_blocks, 1, num_batches);

  constexpr bool fast_path_eligible =
      is_tensor_view_v<OutType> &&
      is_tensor_view_v<InType> &&
      is_tensor_view_v<FilterType>;
  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    // Dispatch on num_channels in [2, FusedChanThreshold]. Generated at
    // compile time from the threshold so raising FusedChanThreshold
    // automatically grows the dispatch table; no per-N switch case to
    // update. A runtime value outside the range falls through to MATX_THROW.
    constexpr int kMinChan = 2;
    [[maybe_unused]] constexpr int kMaxChan = static_cast<int>(FusedChanThreshold);
    const bool matched = [&]<int... Is>(cuda::std::integer_sequence<int, Is...>) {
      return ((num_channels == kMinChan + Is
                 ? (ChannelizePoly1D_FusedChan<THREADS, kMinChan + Is, IsUnitStride,
                                               OutType, InType, FilterType, AccumType>
                        <<<grid, THREADS, 0, stream>>>(o, i, filter),
                    true)
                 : false)
               || ...);
    }(cuda::std::make_integer_sequence<int, kMaxChan - kMinChan + 1>{});
    if (!matched) {
      MATX_THROW(matxInvalidDim, "channelize_poly: channel count not supported with fused kernel");
    }
  };
  if constexpr (fast_path_eligible) {
    const bool is_unit_stride =
        o.Stride(OutType::Rank() - 1) == 1 &&
        i.Stride(InType::Rank() - 1) == 1 &&
        filter.Stride(FilterType::Rank() - 1) == 1;
    if (is_unit_stride) {
      launch(cuda::std::bool_constant<true>{});
    } else {
      launch(cuda::std::bool_constant<false>{});
    }
  } else {
    launch(cuda::std::bool_constant<false>{});
  }
#endif
}

template <typename DataType>
inline void UnpackDFT(DataType inout, cudaStream_t stream)
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

  constexpr bool fast_path_eligible = is_tensor_view_v<DataType>;
  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    ChannelizePoly1DUnpackDFT<IsUnitStride, DataType><<<grid, THREADS, 0, stream>>>(inout);
  };
  if constexpr (fast_path_eligible) {
    const bool is_unit_stride = inout.Stride(DataType::Rank() - 1) == 1;
    if (is_unit_stride) {
      launch(cuda::std::bool_constant<true>{});
    } else {
      launch(cuda::std::bool_constant<false>{});
    }
  } else {
    launch(cuda::std::bool_constant<false>{});
  }
#endif
}

} // end namespace cpoly
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
 * @param num_channels Number of channels in which to separate the signal. Must be positive.
 * num_channels == 1 is the degenerate (no-channelization) case and degrades to a plain FIR filter
 * with decimation_factor == 1.
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

  MATX_ASSERT_STR(num_channels > 0, matxInvalidParameter,
    "channelize_poly: num_channels must be positive");
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
    // num_channels == 1 is degenerate (no channelization, trivial DFT) and
    // the fused kernel's switch starts at N=2. Let num_channels==1 fall
    // through to Smem / SmemTiled / Generic, all of which handle it
    // correctly as a plain FIR.
    if (decimation_factor == num_channels && num_channels >= 2 &&
        num_channels <= detail::cpoly::FusedChanThreshold) {
      detail::cpoly::FusedChan<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
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

      if (decimation_factor == num_channels && detail::cpoly::ShouldUseSmem(out, in, f)) {
        detail::cpoly::Smem<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, stream);
      } else if (detail::cpoly::ShouldUseSmemTiled(out, in, f, decimation_factor)) {
        detail::cpoly::SmemTiled<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, decimation_factor, stream);
      } else {
        detail::cpoly::Generic<decltype(fft_in_slice), InputOp, FilterOp, AccumType>(fft_in_slice, in, f, decimation_factor, stream);
      }
      stop_dims[OUT_RANK-1] = (num_channels/2) + 1;
      auto out_packed = slice<OUT_RANK>(out, start_dims, stop_dims);
      (out_packed = fft(fft_in_slice, num_channels)).run(stream);
      detail::cpoly::UnpackDFT(out, stream);
    }
  } else {
    // The fused-DFT kernel only supports the maximally decimated case (D == M).
    // num_channels == 1 is degenerate (no channelization, trivial DFT) and
    // the fused kernel's switch starts at N=2. Let num_channels==1 fall
    // through to Smem / SmemTiled / Generic, all of which handle it
    // correctly as a plain FIR.
    if (decimation_factor == num_channels && num_channels >= 2 &&
        num_channels <= detail::cpoly::FusedChanThreshold) {
      detail::cpoly::FusedChan<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
    } else {
      if (decimation_factor == num_channels && detail::cpoly::ShouldUseSmem(out, in, f)) {
        detail::cpoly::Smem<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, stream);
      } else if (detail::cpoly::ShouldUseSmemTiled(out, in, f, decimation_factor)) {
        detail::cpoly::SmemTiled<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, decimation_factor, stream);
      } else {
        detail::cpoly::Generic<OutputOp, InputOp, FilterOp, AccumType>(out, in, f, decimation_factor, stream);
      }
      // Specify FORWARD here to prevent any normalization after the ifft. We do not
      // want any extra scaling on the output values.
      (out = ifft(out, num_channels, FFTNorm::FORWARD)).run(stream);
    }
  }
}

/**
 * @brief Host implementation of the 1D polyphase channelizer.
 *
 * This is a feature-parity implementation for CPU executors. It directly
 * computes the per-branch FIR values and then applies the unnormalized,
 * positive-sign DFT used by the CUDA channelizer.
 */
template <typename OutType, typename InType, typename FilterType, typename AccumType, ThreadsMode MODE>
inline void channelize_poly_impl(OutType out, const InType &in, const FilterType &f,
                   index_t num_channels, index_t decimation_factor,
                   [[maybe_unused]] const HostExecutor<MODE> &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using OutputOp = std::remove_cv_t<std::remove_reference_t<OutType>>;
  using InputOp = std::remove_cv_t<std::remove_reference_t<InType>>;
  using FilterOp = std::remove_cv_t<std::remove_reference_t<FilterType>>;
  using input_t = typename InputOp::value_type;
  using filter_t = typename FilterOp::value_type;
  using output_t = typename OutputOp::value_type;
  using filtering_accum_t = cuda::std::conditional_t<
      is_complex_v<input_t> || is_complex_v<filter_t>,
      typename detail::scalar_to_complex<AccumType>::ctype,
      AccumType>;
  using complex_accum_t = typename detail::scalar_to_complex<AccumType>::ctype;

  static_assert(!is_complex_v<AccumType>,
    "channelize_poly: accumulator type must be real; it will be treated as complex when necessary");

  constexpr int IN_RANK = InputOp::Rank();
  constexpr int OUT_RANK = OutputOp::Rank();

  MATX_STATIC_ASSERT_STR(OUT_RANK == IN_RANK+1, matxInvalidDim,
    "channelize_poly: output rank should be 1 higher than input");
  MATX_STATIC_ASSERT_STR(is_complex_v<output_t> || is_complex_half_v<output_t>,
    matxInvalidType, "channelize_poly: output type must be complex");
  MATX_STATIC_ASSERT_STR(FilterType::Rank() == 1, matxInvalidDim,
    "channelize_poly: currently only support 1D filters");

  MATX_ASSERT_STR(num_channels > 0, matxInvalidParameter,
    "channelize_poly: num_channels must be positive");
  MATX_ASSERT_STR(decimation_factor > 0, matxInvalidParameter,
    "channelize_poly: decimation_factor must be positive");
  MATX_ASSERT_STR(decimation_factor <= num_channels, matxInvalidParameter,
    "channelize_poly: decimation_factor must be <= num_channels");

  for (int i = 0; i < IN_RANK-1; i++) {
    MATX_ASSERT_STR(out.Size(i) == in.Size(i), matxInvalidDim,
      "channelize_poly: input/output must have matched batch sizes");
  }

  const index_t input_len = in.Size(IN_RANK-1);
  const index_t num_elem_per_channel = (input_len + decimation_factor - 1) / decimation_factor;
  MATX_ASSERT_STR(out.Size(OUT_RANK-1) == num_channels, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-1 mismatch");
  MATX_ASSERT_STR(out.Size(OUT_RANK-2) == num_elem_per_channel, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-2 mismatch");

  const index_t filter_full_len = f.Size(FilterOp::Rank()-1);
  const index_t filter_phase_len = (filter_full_len + num_channels - 1) / num_channels;
  index_t batch_count = 1;
  for (int i = 0; i < IN_RANK-1; i++) {
    batch_count *= in.Size(i);
  }

  std::vector<complex_accum_t> twiddles(static_cast<size_t>(num_channels * num_channels));
  for (index_t channel = 0; channel < num_channels; channel++) {
    for (index_t branch = 0; branch < num_channels; branch++) {
      twiddles[static_cast<size_t>(channel * num_channels + branch)] =
          detail::cpoly::HostTwiddle<complex_accum_t>(channel, branch, num_channels);
    }
  }

  const index_t num_thread_buffers = std::max<index_t>(1, exec.GetNumThreads());
  std::vector<filtering_accum_t> filtered_storage(
      static_cast<size_t>(num_thread_buffers * num_channels));

  const auto compute_output = [&](index_t batch, index_t t) {
    const auto in_batch_idx = detail::BlockToIdx(in, batch, 1);
    const auto out_batch_idx = detail::BlockToIdx(out, batch, 2);
    index_t thread_index = 0;
#ifdef MATX_EN_OMP
    if (num_thread_buffers > 1) {
      thread_index = static_cast<index_t>(omp_get_thread_num());
    }
#endif
    auto *filtered = filtered_storage.data() +
        static_cast<size_t>(thread_index * num_channels);

    for (index_t branch = 0; branch < num_channels; branch++) {
      filtering_accum_t accum{};
      index_t h_ind = branch;
      index_t sample_idx = 0;
      index_t niter = 0;

      if (decimation_factor == num_channels) {
        const index_t s = num_channels - 1 - branch;
        sample_idx = s + t * num_channels;
        index_t h_skip = 0;
        if (sample_idx >= input_len) {
          h_skip = 1;
          sample_idx -= num_channels;
        }

        index_t available_taps = filter_phase_len;
        if (filter_phase_len > 0 &&
            ((filter_phase_len - 1) * num_channels + branch) >= filter_full_len) {
          available_taps--;
        }

        if (available_taps > h_skip && (t + 1) > h_skip) {
          niter = std::min(available_taps - h_skip, t + 1 - h_skip);
          h_ind = branch + h_skip * num_channels;
        }
      } else {
        const index_t r_remapped = (branch + num_channels - decimation_factor) % num_channels;
        const index_t s = num_channels - 1 - r_remapped;
        const index_t last_arrived = t * decimation_factor + decimation_factor - 1;
        if (last_arrived >= s) {
          const index_t A = last_arrived - s;
          sample_idx = last_arrived - (A % num_channels);
          const index_t causal_count = A / num_channels + 1;
          const index_t phase = (branch + t * decimation_factor) % num_channels;
          index_t h_skip = 0;
          if (sample_idx >= input_len) {
            h_skip = 1;
            sample_idx -= num_channels;
          }

          index_t available_taps = filter_phase_len;
          if (filter_phase_len > 0 &&
              ((filter_phase_len - 1) * num_channels + phase) >= filter_full_len) {
            available_taps--;
          }

          if (available_taps > h_skip && causal_count > h_skip) {
            niter = std::min(available_taps - h_skip, causal_count - h_skip);
            h_ind = phase + h_skip * num_channels;
          }
        }
      }

      for (index_t i = 0; i < niter; i++) {
        const input_t in_val = detail::cpoly::HostReadSignal(in, in_batch_idx, sample_idx);
        const filter_t h_val = f(h_ind);
        detail::cpoly::HostChannelizeCmac(accum,
          detail::cpoly::HostChannelizeCastFilter<filtering_accum_t>(h_val),
          detail::cpoly::HostChannelizeCastInput<filtering_accum_t>(in_val));
        h_ind += num_channels;
        sample_idx -= num_channels;
      }

      filtered[static_cast<size_t>(branch)] = accum;
    }

    for (index_t channel = 0; channel < num_channels; channel++) {
      complex_accum_t dft{};
      for (index_t branch = 0; branch < num_channels; branch++) {
        dft += detail::cpoly::HostAsComplex<complex_accum_t>(
            filtered[static_cast<size_t>(branch)]) *
            twiddles[static_cast<size_t>(channel * num_channels + branch)];
      }
      detail::cpoly::HostWriteOutput(out, out_batch_idx, t, channel, dft);
    }
  };

  const index_t total_outputs = batch_count * num_elem_per_channel;
#ifdef MATX_EN_OMP
  if (exec.GetNumThreads() > 1) {
    #pragma omp parallel for num_threads(exec.GetNumThreads())
    for (index_t i = 0; i < total_outputs; i++) {
      compute_output(i / num_elem_per_channel, i % num_elem_per_channel);
    }
  } else
#endif
  {
    for (index_t i = 0; i < total_outputs; i++) {
      compute_output(i / num_elem_per_channel, i % num_elem_per_channel);
    }
  }
}
} // end namespace matx
