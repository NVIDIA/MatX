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


// Number of output samples per channel per iteration for the kernel that stores
// the input data in shared memory. Ideally, this value would be determined dynamically
// to balance occupancy and CTA size. For now, we choose a reasonable default.
constexpr index_t FullSmemKernelNoutPerIter = 4;

// Maximum dynamic shared memory (bytes) for caching filter taps in ChannelizePoly1D.
constexpr size_t GenericMaxFilterSmemBytes = 6 * 1024;

// Constants for the tiled shared-memory kernel, which processes CTILE channels
// per CTA. NOUT=16 is the maximally decimated tile: four time lanes each
// compute four rows, reusing every filter tap across those rows. NOUT=4 is the
// oversampled tile and the critical fallback when the grouped ring does not fit.
constexpr int SmemTiledCtile = 32;
constexpr int SmemTiledNout = 4;
constexpr int SmemTiledMaxDecNout = 16;

// Dispatch policy. Up to this many channels, a FIR that is not fused uses the
// whole-channel Smem kernel (critical) or the Generic kernel (oversampled);
// wider channel counts use the channel-tiled kernel.
constexpr index_t SmallChannelLimit = 16;
// Oversampled tiles cache the Full filter layout only up to this size. Every CTA stages the whole
// filter, so larger filters cost more to load than they save and reduce occupancy.
constexpr size_t SmemTiledOversampledFilterBytes = 4 * 1024;

// Memory bus width per SM at or above this marks an HBM-class GPU (roughly
// 40-55 bits per SM, versus about 3-16 for GDDR and LPDDR parts).
constexpr int HighBandwidthBusBitsPerSm = 32;

// Device attributes used by dispatch and launch sizing. Each is queried at most
// once per channelizer call, and only when a decision needs it.
class DeviceAttrs {
public:
  int SmCount() { return Get(sms_, cudaDevAttrMultiProcessorCount); }
  size_t L2Bytes()
  {
    return static_cast<size_t>(Get(l2_bytes_, cudaDevAttrL2CacheSize));
  }
  // GPUs whose FP64 rate is far below FP32 make FP64 channelizers arithmetic
  // bound, so padded or idle lanes cost time directly.
  bool LowFp64Throughput()
  {
    return Get(fp64_ratio_, cudaDevAttrSingleToDoublePrecisionPerfRatio) > 8;
  }
  // Bus width is a cheap proxy for bandwidth; memory clock queries are slow.
  bool HighMemoryBandwidth()
  {
    return Get(bus_bits_, cudaDevAttrGlobalMemoryBusWidth) >= HighBandwidthBusBitsPerSm * SmCount();
  }

private:
  static int Get(int &value, cudaDeviceAttr attr)
  {
    if (value < 0) value = GetDeviceAttr(attr);
    return value;
  }
  int sms_ = -1;
  int l2_bytes_ = -1;
  int fp64_ratio_ = -1;
  int bus_bits_ = -1;
};

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

// A tiled launch computes SmemTiledCtile channels by nout output rows per
// iteration: SmemTiledMaxDecNout (critical only) or SmemTiledNout.
struct SmemTiledPlan {
  int nout;
  SmemTiledFilterLayout filter_layout;
  size_t bytes;
};

// Tiled grids target a number of CTAs per SM so they scale with the GPU.
constexpr int SmemTiledCtasPerSm = 8;

constexpr int SmemTiledChannelTiles(index_t num_channels)
{
  return static_cast<int>((num_channels + SmemTiledCtile - 1) / SmemTiledCtile);
}

// A tiled plan launches when its shared memory fits the default limit and its
// channel tiles fit the grid.y limit.
inline bool SmemTiledFits(const SmemTiledPlan &plan, index_t num_channels)
{
  return plan.bytes <= MaxDefaultDynamicSmemBytes &&
      SmemTiledChannelTiles(num_channels) <= 65535;
}

// Rows per CTA for the tiled kernel. A single batch spreads target_ctas over
// its channel tiles; batches share the grid, bounding the rows each CTA adds.
inline index_t SmemTiledElemsPerBlock(
    index_t nout_per_channel, index_t num_channels, index_t batches, index_t target_ctas)
{
  const index_t channel_tiles = SmemTiledChannelTiles(num_channels);
  const index_t time_blocks = (target_ctas + channel_tiles - 1) / channel_tiles;
  const index_t single = (nout_per_channel + time_blocks - 1) / time_blocks;
  const index_t spatial_batches = std::max<index_t>(1, channel_tiles * batches);
  const index_t time_targets = std::max<index_t>(1,
      (2 * target_ctas + spatial_batches - 1) / spatial_batches);
  const index_t span = (nout_per_channel + time_targets - 1) / time_targets;
  return std::max(single, std::min<index_t>(span, 128));
}

// Whether 32-bit indices can address a tiled launch whose window ends before
// global output row window_end. The input ring loads through the end of the row
// holding the window's newest sample, which can pass input_len + M when D < M.
inline bool SmemTiledFitsInt32(
    index_t input_len, index_t num_channels, index_t decimation_factor, index_t window_end)
{
  constexpr int64_t max_index = std::numeric_limits<int32_t>::max();
  const int64_t newest_row =
      (static_cast<int64_t>(window_end) * decimation_factor - 1) / num_channels;
  return static_cast<int64_t>(input_len) + num_channels <= max_index &&
      (newest_row + 1) * num_channels - 1 <= max_index;
}

// The Rotated layout stores K = M / gcd(M, D) phases per tile channel; the
// kernel supports at most SmemTiledMaxRotations of them.
inline bool SmemTiledRotatedLayoutSupported(index_t num_channels, index_t decimation_factor)
{
  if (decimation_factor == num_channels) {
    return true;
  }
  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  return num_channels / gcd_val <= SmemTiledMaxRotations;
}

// Compute the shared memory footprint for the SmemTiled filter taps in the
// Rotated layout (one [channel][k][p] block per CTILE, K phases duplicated
// across channels, zero-padded when CTILE > M).
// For D==M: one phase per tile channel -> CTILE * P elements.
// For D<M: K phases per tile channel -> CTILE * K * P elements.
template <typename FilterType>
inline size_t SmemTiledFilterBytesRotated(
    index_t num_channels, index_t filter_len, index_t decimation_factor)
{
  using filter_t = typename FilterType::value_type;
  const index_t P = (filter_len + num_channels - 1) / num_channels;
  if (decimation_factor == num_channels) {
    return static_cast<size_t>(SmemTiledCtile) * P * sizeof(filter_t);
  }
  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const index_t K = num_channels / gcd_val;
  return static_cast<size_t>(SmemTiledCtile) * K * P * sizeof(filter_t);
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

template <typename InType>
inline size_t SmemTiledInputBytes(
    index_t num_channels, index_t filter_len, index_t decimation, int nout)
{
  const index_t P = (filter_len + num_channels - 1) / num_channels;
  return static_cast<size_t>(
      SmemTiledInputHeight(P, num_channels, decimation, nout)) * SmemTiledCtile *
      sizeof(typename InType::value_type);
}

// Size an explicit tile height and filter layout. The plan is launchable when
// bytes <= MaxDefaultDynamicSmemBytes and, for Rotated, the rotation count is supported.
template <typename OutType, typename InType, typename FilterType>
inline SmemTiledPlan SmemTiledPlanWithLayout(
    const OutType &o, const InType &, const FilterType &filter,
    index_t decimation_factor, int nout, SmemTiledFilterLayout layout)
{
  using input_t = typename InType::value_type;
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const index_t filter_len = filter.Size(FilterType::Rank() - 1);
  size_t filter_bytes = 0;
  if (layout == SmemTiledFilterLayout::Full) {
    filter_bytes = SmemTiledFilterBytesFull<FilterType>(num_channels, filter_len);
  } else if (layout == SmemTiledFilterLayout::Rotated) {
    filter_bytes = SmemTiledFilterBytesRotated<FilterType>(
        num_channels, filter_len, decimation_factor);
  }
  return {nout, layout,
      SmemTiledInputBytes<InType>(num_channels, filter_len, decimation_factor, nout) +
      MATX_ROUND_UP(filter_bytes, sizeof(input_t))};
}

// Critical tiles cache the smaller of the Full and Rotated layouts that fits, falling back from the
// grouped 32x16 tile to 32x4 when its ring does not fit. Oversampled tiles use 32x4 and cache only
// small Full filters. Global keeps the filter in device memory. The caller checks SmemTiledFits and
// uses Generic when no tile fits.
template <typename OutType, typename InType, typename FilterType>
inline SmemTiledPlan SelectTiledPlan(
    const OutType &o, const InType &in, const FilterType &filter, index_t decimation_factor)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  auto plan_for = [&](int nout, SmemTiledFilterLayout layout) {
    return SmemTiledPlanWithLayout(o, in, filter, decimation_factor, nout, layout);
  };
  if (decimation_factor == num_channels) {
    SmemTiledPlan global{};
    for (int nout : {SmemTiledMaxDecNout, SmemTiledNout}) {
      const auto full = plan_for(nout, SmemTiledFilterLayout::Full);
      const auto rotated = plan_for(nout, SmemTiledFilterLayout::Rotated);
      const auto &cached = full.bytes <= rotated.bytes ? full : rotated;
      if (cached.bytes <= MaxDefaultDynamicSmemBytes) return cached;
      global = plan_for(nout, SmemTiledFilterLayout::Global);
      if (global.bytes <= MaxDefaultDynamicSmemBytes) return global;
    }
    return global;
  }
  const auto full = plan_for(SmemTiledNout, SmemTiledFilterLayout::Full);
  const size_t filter_bytes = SmemTiledFilterBytesFull<FilterType>(
      num_channels, filter.Size(FilterType::Rank() - 1));
  if (filter_bytes <= SmemTiledOversampledFilterBytes && full.bytes <= MaxDefaultDynamicSmemBytes) {
    return full;
  }
  return plan_for(SmemTiledNout, SmemTiledFilterLayout::Global);
}

template <typename Launch, typename... Ops>
inline void DispatchUnitStride(Launch &&launch, const Ops &...ops)
{
  if constexpr ((is_tensor_view_v<Ops> && ...)) {
    if (((ops.Stride(Ops::Rank() - 1) == 1) && ...)) {
      launch(cuda::std::bool_constant<true>{});
      return;
    }
  }
  launch(cuda::std::bool_constant<false>{});
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

template <typename AccumType, typename OutType, typename InType, typename FilterType>
inline void ValidateChannelizePolyArgs(
    [[maybe_unused]] const OutType &out, const InType &in, const FilterType &,
    [[maybe_unused]] index_t num_channels, index_t decimation_factor,
    [[maybe_unused]] index_t out_elem_offset)
{
  using output_t = typename OutType::value_type;
  constexpr int IN_RANK = InType::Rank();
  constexpr int OUT_RANK = OutType::Rank();

  static_assert(!is_complex_v<AccumType>,
      "channelize_poly: accumulator type must be real; "
      "it will be treated as complex when necessary");
  MATX_STATIC_ASSERT_STR(OUT_RANK == IN_RANK + 1, matxInvalidDim,
      "channelize_poly: output rank should be 1 higher than input");
  MATX_STATIC_ASSERT_STR(is_complex_v<output_t>, matxInvalidType,
      "channelize_poly: output type must be complex");
  MATX_STATIC_ASSERT_STR(FilterType::Rank() == 1, matxInvalidDim,
      "channelize_poly: currently only support 1D filters");

  MATX_ASSERT_STR(num_channels > 0, matxInvalidParameter,
      "channelize_poly: num_channels must be positive");
  MATX_ASSERT_STR(decimation_factor > 0, matxInvalidParameter,
      "channelize_poly: decimation_factor must be positive");
  MATX_ASSERT_STR(decimation_factor <= num_channels, matxInvalidParameter,
      "channelize_poly: decimation_factor must be <= num_channels");

  for (int i = 0; i < IN_RANK - 1; i++) {
    MATX_ASSERT_STR(out.Size(i) == in.Size(i), matxInvalidDim,
        "channelize_poly: input/output must have matched batch sizes");
  }

  [[maybe_unused]] const index_t num_elem_per_channel =
      (in.Size(IN_RANK - 1) + decimation_factor - 1) / decimation_factor;
  MATX_ASSERT_STR(out.Size(OUT_RANK - 1) == num_channels, matxInvalidDim,
      "channelize_poly: output size OUT_RANK-1 mismatch");
  // A window may cover any part of the full output-element grid, including a
  // prefix beginning at zero that is shorter than the full grid.
  MATX_ASSERT_STR(
      out.Size(OUT_RANK - 2) + out_elem_offset <= num_elem_per_channel, matxInvalidDim,
      "channelize_poly: output-element window exceeds the full output size");
}

template <int NOUT, bool MaximallyDecimated,
          typename OutType, typename InType, typename FilterType, typename AccumType>
inline void SmemTiledImpl(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, const SmemTiledPlan &plan,
    cudaStream_t stream, index_t out_elem_offset, DeviceAttrs &attrs)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  constexpr int CTILE = SmemTiledCtile;

  const index_t num_channels = o.Size(OutType::Rank() - 1);
  const index_t nout_per_channel = o.Size(OutType::Rank() - 2);
  const int num_batches = static_cast<int>(TotalSize(i) / i.Size(i.Rank() - 1));

  const index_t gcd_val = std::gcd(num_channels, decimation_factor);
  const int32_t K = static_cast<int32_t>(num_channels / gcd_val);

  const int channel_tiles = SmemTiledChannelTiles(num_channels);
  // Target spatial blocks across the whole batch, with bounded CTA work.
  const index_t target_ctas = SmemTiledCtasPerSm * attrs.SmCount();
  int elem_per_block = static_cast<int>(SmemTiledElemsPerBlock(
      nout_per_channel, num_channels, num_batches, target_ctas));
  // Use the full time dimension of oversampled FIR tiles when possible.
  if (decimation_factor < num_channels) {
    elem_per_block = std::max(elem_per_block, NOUT);
  }
  if constexpr (MaximallyDecimated && NOUT > SmemTiledMaxDecYThreads) {
    // Each iteration computes a full group even in a partial CTA. Pack output
    // rows into complete groups so only the final CTA can waste that work.
    elem_per_block = ((elem_per_block + NOUT - 1) / NOUT) * NOUT;
  }
  const int time_blocks = static_cast<int>(
      (nout_per_channel + elem_per_block - 1) / elem_per_block);

  dim3 block(CTILE, MaximallyDecimated ? SmemTiledMaxDecYThreads : NOUT);
  dim3 grid(time_blocks, channel_tiles, num_batches);

  // Use int32_t for intra-kernel index arithmetic when all indices fit,
  // avoiding 64-bit IMAD.WIDE instructions in the inner loops.
  const bool use_32bit = (sizeof(index_t) <= sizeof(int32_t)) ||
      SmemTiledFitsInt32(i.Size(i.Rank() - 1), num_channels, decimation_factor,
                         out_elem_offset + nout_per_channel);

  // Dispatch on MaximallyDecimated x (FilterInSmem, FilterFullLayout) x
  // IndexType x IsUnitStride. Filter-smem layout selection: Full (smallest
  // footprint, phase compute at access), Rotated (direct indexing, redundant
  // storage), or Global (filter stays in GMEM).
  auto launch = [&](auto idx_tag, auto is_unit_c) {
    using IdxT = decltype(idx_tag);
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    const IdxT epb = static_cast<IdxT>(elem_per_block);
    const IdxT df  = static_cast<IdxT>(decimation_factor);
    const IdxT oeo = static_cast<IdxT>(out_elem_offset);

    auto launch_with_layout = [&](auto in_smem_c, auto full_c) {
      constexpr bool FIS = decltype(in_smem_c)::value;
      constexpr bool FFL = decltype(full_c)::value;
      ChannelizePoly1D_SmemTiled<CTILE, NOUT, MaximallyDecimated,
          FIS, FFL, IsUnitStride, IdxT,
          OutType, InType, FilterType, AccumType>
          <<<grid, block, plan.bytes, stream>>>(o, i, filter, epb, df, K, oeo);
    };

    switch (plan.filter_layout) {
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
    DispatchUnitStride([&](auto is_unit_c) {
      launch(idx_tag, is_unit_c);
    }, o, i, filter);
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

// Launch the tile height and filter layout selected by the shared-memory fit check.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void SmemTiled(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, const SmemTiledPlan &plan,
    cudaStream_t stream, index_t out_elem_offset, DeviceAttrs &attrs)
{
  const index_t num_channels = o.Size(OutType::Rank() - 1);
  auto launch = [&]<int NOUT, bool MaximallyDecimated>() {
    SmemTiledImpl<NOUT, MaximallyDecimated, OutType, InType, FilterType, AccumType>(
        o, i, filter, decimation_factor, plan, stream, out_elem_offset, attrs);
  };
  if (plan.nout == SmemTiledMaxDecNout && decimation_factor == num_channels) {
    launch.template operator()<SmemTiledMaxDecNout, true>();
  } else if (plan.nout == SmemTiledNout) {
    // Critical 32x4 fallbacks reuse the oversampled leaf's K=1 case.
    launch.template operator()<SmemTiledNout, false>();
  } else {
    MATX_THROW(matxInvalidParameter, "channelize_poly: unsupported tiled row count");
  }
}

// out_elem_offset selects a window of the full per-channel output-element (time)
// grid: the output tensor `o` is sized to the window length and receives the
// output elements whose global time indices are
// [out_elem_offset, out_elem_offset + o.Size(OutRank-2)). The streaming
// channelizer uses this to compute only the elements that will be output based
// on newly provided samples versus the full grid.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void Generic(OutType o, const InType &i,
                                     const FilterType &filter, index_t decimation_factor, cudaStream_t stream,
                                     index_t out_elem_offset = 0)
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
  const index_t elem_blocks = (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  // Channels vary fastest (grid.x) for input reuse through L2. Time blocks use
  // grid.y, split into launches of at most 65535 blocks.
  constexpr index_t max_grid_y = 65535;
  MATX_ASSERT_STR(elem_blocks <= std::numeric_limits<int>::max(), matxInvalidSize,
      "channelize_poly: output too long for the generic kernel");

  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    for (index_t offset = 0; offset < elem_blocks; offset += max_grid_y) {
      const dim3 grid(static_cast<uint32_t>(num_channels),
                      static_cast<uint32_t>(std::min(max_grid_y, elem_blocks - offset)),
                      num_batches);
      if (decimation_factor == num_channels) {
        // For M == D, cache one filter phase in dynamic shared memory if it fits.
        const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;
        const size_t smem_needed = static_cast<size_t>(filter_phase_len) * sizeof(filter_t);
        const uint32_t smem_bytes = (smem_needed <= GenericMaxFilterSmemBytes)
            ? static_cast<uint32_t>(smem_needed) : 0;
        ChannelizePoly1D<THREADS, true, IsUnitStride, OutType, InType, FilterType, AccumType>
            <<<grid, THREADS, smem_bytes, stream>>>(
                o, i, filter, decimation_factor, smem_bytes, out_elem_offset,
                static_cast<int>(offset));
      } else {
        ChannelizePoly1D<THREADS, false, IsUnitStride, OutType, InType, FilterType, AccumType>
            <<<grid, THREADS, 0, stream>>>(
                o, i, filter, decimation_factor, 0, out_elem_offset, static_cast<int>(offset));
      }
    }
  };

  DispatchUnitStride(launch, o, i, filter);
#endif
}

template <typename OutType, typename InType, typename FilterType>
inline size_t SmemSizeBytes(
    const OutType &o, const InType &, const FilterType &filter,
    int nout = FullSmemKernelNoutPerIter)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;

  index_t filter_len = filter.Size(FilterType::Rank()-1);

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t filter_phase_len = (filter_len + num_channels - 1) / num_channels;

  size_t smem_size = sizeof(filter_t)*(num_channels)*(filter_phase_len) +
    sizeof(input_t)*(num_channels)*(filter_phase_len + nout - 1);
  const size_t max_sizeof = cuda::std::max(sizeof(filter_t), sizeof(input_t));
  if (smem_size % max_sizeof) {
    smem_size += max_sizeof - (smem_size % max_sizeof);
  }
  return smem_size;
}

template <typename OutType, typename InType, typename FilterType>
inline bool ShouldUseSmem(const OutType &out, const InType &in, const FilterType &filter)
{
  // The full shared memory kernel uses blocks of size
  // (num_channels, FullSmemKernelNoutPerIter), so ensure
  // that the resulting thread per block count will not exceed MAX_NUM_THREADS_PER_BLOCK
  const int MAX_NUM_THREADS_PER_BLOCK = 1024;
  const index_t num_channels = out.Size(OutType::Rank()-1);
  return (
      SmemSizeBytes(out, in, filter) <= MaxDefaultDynamicSmemBytes &&
      num_channels <= (MAX_NUM_THREADS_PER_BLOCK/FullSmemKernelNoutPerIter));
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void Smem(OutType o, const InType &i, const FilterType &filter,
                 cudaStream_t stream, index_t out_elem_offset, DeviceAttrs &attrs)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  dim3 block(static_cast<int>(num_channels), FullSmemKernelNoutPerIter);
  size_t smem_size = SmemSizeBytes(o, i, filter);

  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    auto kernel = ChannelizePoly1D_Smem<IsUnitStride, OutType, InType, FilterType, AccumType>;
    const index_t sms = attrs.SmCount();
    const index_t l2_bytes = static_cast<index_t>(attrs.L2Bytes());
    const index_t row_bytes = num_channels * static_cast<index_t>(
        sizeof(typename InType::value_type) + 2 * sizeof(AccumType));
    const size_t batch_bytes = static_cast<size_t>(nout_per_channel) * row_bytes;
    const size_t input_bytes = static_cast<size_t>(TotalSize(i)) *
        sizeof(typename InType::value_type);
    // When input fits L2 but the pipeline footprint does not, retain small
    // whole-warp tiles and enough CTAs to cover the kernel's residency.
    const bool cache_pressure = block.x * block.y <= 3 * 32 &&
        input_bytes < static_cast<size_t>(l2_bytes) && batch_bytes >= static_cast<size_t>(l2_bytes);
    // Taller groups amortize barriers in blocks with fewer than four warps.
    // Keep at least one CTA per SM and preserve the original fit fallback.
    while (block.y < 16 && block.x * block.y <= 3 * 32 &&
           (!cache_pressure || (block.x * block.y) % 32 != 0)) {
      const int next = static_cast<int>(2 * block.y);
      const size_t bytes = SmemSizeBytes(o, i, filter, next);
      if (bytes > MaxDefaultDynamicSmemBytes ||
          ((nout_per_channel + next - 1) / next) * num_batches < sms)
        break;
      block.y = static_cast<uint32_t>(next);
      smem_size = bytes;
    }
    const index_t group = block.y;
    int resident_blocks = 0;
    MATX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, kernel, static_cast<int>(block.x * block.y), smem_size));
    index_t targets = std::max<index_t>(1,
        ((cache_pressure ? 1 : 2) * sms * resident_blocks + num_batches - 1) /
            num_batches);
    index_t elem_per_block = (nout_per_channel + targets - 1) / targets;
    if (group > FullSmemKernelNoutPerIter)
      elem_per_block = std::max(elem_per_block, group);
    // Bound serial work without rounding longer spans and reducing coverage.
    // This is a logical work budget, not exclusive L2 ownership.
    if (sms > 0 && l2_bytes > 0) {
      const index_t limit = std::max(group, (l2_bytes / sms / row_bytes / group) * group);
      elem_per_block = std::min(elem_per_block, limit);
    }
    // Amortize a mostly idle final iteration without another tile shape.
    const index_t rounded = MATX_ROUND_UP(elem_per_block, group);
    if (group > FullSmemKernelNoutPerIter &&
        elem_per_block >= group && 4 * elem_per_block < 3 * rounded) {
      elem_per_block = rounded;
    }
    dim3 grid(static_cast<uint32_t>(
        (nout_per_channel + elem_per_block - 1) / elem_per_block), 1, num_batches);
    kernel
        <<<grid, block, smem_size, stream>>>(o, i, filter, elem_per_block, out_elem_offset);
  };
  DispatchUnitStride(launch, o, i, filter);
#endif
}

template <int NUM_CHAN, typename OutType, typename InType, typename FilterType, typename AccumType>
inline size_t FusedRadixSizeBytes(
    const OutType &out, const InType &, const FilterType &filter, index_t decimation_factor)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  using critical_layout = FusedRadixSmemLayout<NUM_CHAN, true, input_t, filter_t, AccumType>;
  using oversampled_layout = FusedRadixSmemLayout<NUM_CHAN, false, input_t, filter_t, AccumType>;
  const index_t P = (filter.Size(FilterType::Rank() - 1) + NUM_CHAN - 1) / NUM_CHAN;
  const size_t bytes = decimation_factor == NUM_CHAN
      ? critical_layout(P, decimation_factor).bytes
      : oversampled_layout(P, decimation_factor).bytes;

  if constexpr (NUM_CHAN >= 3 && NUM_CHAN <= 6) {
    if (decimation_factor == NUM_CHAN &&
        FusedSmallUseDirect<NUM_CHAN, OutType, InType, FilterType, AccumType>(
            P, out.Size(OutType::Rank() - 2))) {
      return 0;
    }
  }
  return bytes;
}

// Whether the fused leaf's working set fits the default shared-memory limit.
template <int NUM_CHAN, typename OutType, typename InType, typename FilterType, typename AccumType>
inline bool FusedRadixFits(
    const OutType &out, const InType &in, const FilterType &filter, index_t decimation_factor)
{
  return FusedRadixSizeBytes<NUM_CHAN, OutType, InType, FilterType, AccumType>(
      out, in, filter, decimation_factor) <= MaxDefaultDynamicSmemBytes;
}

// Fused leaves avoid the intermediate filtered tensor and the separate FFT,
// and are preferred whenever they apply. The exceptions are grouped leaves
// (M > 6) that measure slower than the FIR plus cuFFT path:
//  - complex inputs to leaves whose DFT stages go through shared memory (no
//    register FFT) or, when critically sampled, that pad each row to more
//    than twice the channel count. The separate path writes and rereads the
//    filtered intermediate, which costs less than this extra arithmetic while
//    the intermediate stays in L2 or memory bandwidth per SM is high.
//    Oversampling enlarges the intermediate that fusion removes, which
//    outweighs the padding.
//  - FP64 on GPUs with low FP64 throughput, where the fused leaf's padded,
//    complex-valued arithmetic dominates. Complex double always takes the
//    separate path there. Real double does so only when the separate path's
//    working set stays in L2 and each batch is long enough for the FIR and FFT
//    kernels to run efficiently; short or DRAM-bound calls still fuse.
constexpr index_t SeparateFp64MinInputLength = 128 * 1024;

template <int NUM_CHAN, typename InputType, typename AccumType>
inline bool FusedRadixPreferred(index_t decimation_factor, index_t input_length,
                                size_t input_bytes, DeviceAttrs &attrs)
{
  if constexpr (NUM_CHAN <= 6) {
    return true;
  } else if constexpr (!is_complex_v<InputType>) {
    if constexpr (cuda::std::is_same_v<AccumType, double>) {
      return !(input_length >= SeparateFp64MinInputLength &&
               attrs.LowFp64Throughput() && input_bytes <= attrs.L2Bytes());
    }
    return true;
  } else {
    using Config = FusedRadixConfig<NUM_CHAN, AccumType>;
    if (!Config::WarpFft || (decimation_factor == NUM_CHAN && Config::RowStride > 2 * NUM_CHAN)) {
      if (input_bytes <= attrs.L2Bytes() || attrs.HighMemoryBandwidth()) {
        return false;
      }
    }
    if constexpr (cuda::std::is_same_v<AccumType, double>) {
      return !attrs.LowFp64Throughput();
    }
    return true;
  }
}

template <int NUM_CHAN, bool MaximallyDecimated, typename OutType,
          typename InType, typename FilterType, typename AccumType>
inline void FusedRadixImpl(
    OutType o, const InType &i, const FilterType &filter,
    index_t decimation_factor, cudaStream_t stream, index_t out_elem_offset = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  using config = FusedRadixConfig<
      NUM_CHAN, AccumType, MaximallyDecimated, typename InType::value_type>;
  constexpr int threads = config::Threads;
  constexpr int nrows = config::NRows;
  const index_t nout_per_channel = o.Size(OutType::Rank() - 2);
  const int num_batches = static_cast<int>(TotalSize(i) / i.Size(i.Rank() - 1));
  constexpr int target_blocks = 512;
  index_t elem_per_block;
  if constexpr (NUM_CHAN == 2 || (NUM_CHAN <= 6 && MaximallyDecimated)) {
    constexpr int block_rows = threads * config::SmallOutputsPerThread;
    elem_per_block = block_rows;
    if constexpr (NUM_CHAN == 2 && MaximallyDecimated &&
                  cuda::std::is_same_v<AccumType, float> &&
                  is_complex_v<typename InType::value_type>) {
      using traits = ChannelizePolyFusedSmallTraits<
          NUM_CHAN, OutType, InType, FilterType, AccumType>;
      if constexpr (traits::UseM2PairLeaf) {
        // Favor grid coverage for short inputs; keep reuse for larger batches.
        const index_t blocks = (nout_per_channel + block_rows - 1) / block_rows;
        if (blocks <= (target_blocks - 1) / num_batches) elem_per_block = threads;
      }
    }
  } else {
    const index_t target_elems = (nout_per_channel + target_blocks - 1) / target_blocks;
    // Each loop iteration computes a complete NROWS tile. Rounding the CTA's span prevents hundreds
    // of CTAs from doing mostly-invalid tile work for short signals or large channel counts.
    elem_per_block = MATX_ROUND_UP(target_elems, nrows);
  }
  const int time_blocks = static_cast<int>(
      (nout_per_channel + elem_per_block - 1) / elem_per_block);
  const dim3 grid(time_blocks, 1, num_batches);
  const size_t smem_size =
      FusedRadixSizeBytes<NUM_CHAN, OutType, InType, FilterType, AccumType>(
          o, i, filter, decimation_factor);

  auto launch = [&](auto is_unit_c) {
    constexpr bool is_unit_stride = decltype(is_unit_c)::value;
    auto launch_indexed = [&](auto idx_tag) {
      using IdxT = decltype(idx_tag);
      ChannelizePoly1D_FusedRadixPow2<
          threads, NUM_CHAN, nrows, MaximallyDecimated, is_unit_stride,
          OutType, InType, FilterType, AccumType, IdxT>
          <<<grid, threads, smem_size, stream>>>(o, i, filter,
              static_cast<IdxT>(elem_per_block), static_cast<IdxT>(decimation_factor),
              static_cast<IdxT>(out_elem_offset));
    };
    // Narrow only grouped mixed-radix warp FFTs; retain all other index paths.
    if constexpr (sizeof(index_t) <= sizeof(int32_t) || !config::WarpFft || config::Radix1 == 1) {
      launch_indexed(index_t{});
    } else {
      // Keep strides and batch addressing wide; reserve padded-group headroom.
      const index_t padding_rows = elem_per_block + nrows;
      const index_t max_index = std::numeric_limits<int32_t>::max();
      const index_t input_len = i.Size(InType::Rank() - 1);
      const index_t filter_len = filter.Size(FilterType::Rank() - 1);
      const bool padding_fits = padding_rows < max_index / NUM_CHAN;
      const index_t limit = padding_fits ? max_index - padding_rows * NUM_CHAN : 0;
      const bool use_32bit = padding_fits && input_len <= limit &&
          filter_len <= max_index - NUM_CHAN && nout_per_channel <= limit / decimation_factor &&
          out_elem_offset <= limit / decimation_factor - nout_per_channel;
      if (use_32bit) {
        launch_indexed(int32_t{});
      } else {
        launch_indexed(index_t{});
      }
    }
  };
  if constexpr (NUM_CHAN <= 6 && MaximallyDecimated) {
    DispatchUnitStride(launch, o, i, filter);
  } else {
    // General fused launches instantiate only unit-stride tensor paths. The public
    // dispatcher checks this precondition.
    auto check_stride = [](const auto &op) {
      using op_t = cuda::std::remove_cvref_t<decltype(op)>;
      if constexpr (is_tensor_view_v<op_t>) {
        if (op.Stride(op_t::Rank() - 1) != 1) {
          MATX_THROW(matxInvalidParameter,
              "channelize_poly: general fused launch requires unit last-dimension stride");
        }
      }
    };
    check_stride(o);
    check_stride(i);
    check_stride(filter);
    launch(cuda::std::bool_constant<true>{});
  }
#endif
}

// Type-level eligibility of the fused FIR+DFT kernels. Critical M=2..6
// leaves accept expressions, strided views, half precision, and mixed types.
// The grouped leaves require matched float/double types and a tensor filter.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
struct FusedRadixTypes {
  using output_t = typename OutType::value_type;
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  // Input expressions are evaluated while the fused kernel stages its input
  // tile. This preserves narrow source loads and avoids materializing results
  // such as converted, phase-corrected complex samples.
  static constexpr bool Small =
      is_tensor_view_v<OutType> && is_matx_op<InType>() &&
      is_matx_op<FilterType>() && is_complex_v<output_t> &&
      (cuda::std::is_same_v<AccumType, float> ||
       cuda::std::is_same_v<AccumType, double> || is_matx_half_v<AccumType>);
  static constexpr bool General =
      is_tensor_view_v<OutType> && is_matx_op<InType>() && is_tensor_view_v<FilterType> &&
      (cuda::std::is_same_v<AccumType, float> ||
       cuda::std::is_same_v<AccumType, double>) &&
      cuda::std::is_same_v<output_t, cuda::std::complex<AccumType>> &&
      (cuda::std::is_same_v<input_t, AccumType> ||
       cuda::std::is_same_v<input_t, cuda::std::complex<AccumType>>) &&
      cuda::std::is_same_v<filter_t, AccumType>;
};

// Invoke fn with the compile-time channel count of a fused leaf and return its result, or return
// false when no leaf is instantiated for num_channels. M=2...6 replace the older FusedChan family;
// the remaining entries are the measured power-of-two and K-by-power-of-two cases.
template <bool SmallOnly, typename Fn>
inline bool VisitFusedRadixChannels(index_t num_channels, Fn &&fn)
{
  switch (num_channels) {
  case 2: return fn(cuda::std::integral_constant<int, 2>{});
  case 3: return fn(cuda::std::integral_constant<int, 3>{});
  case 4: return fn(cuda::std::integral_constant<int, 4>{});
  case 5: return fn(cuda::std::integral_constant<int, 5>{});
  case 6: return fn(cuda::std::integral_constant<int, 6>{});
  default: break;
  }
  if constexpr (!SmallOnly) {
    switch (num_channels) {
    case 8: return fn(cuda::std::integral_constant<int, 8>{});
    case 10: return fn(cuda::std::integral_constant<int, 10>{});
    case 16: return fn(cuda::std::integral_constant<int, 16>{});
    case 20: return fn(cuda::std::integral_constant<int, 20>{});
    case 32: return fn(cuda::std::integral_constant<int, 32>{});
    case 40: return fn(cuda::std::integral_constant<int, 40>{});
    case 64: return fn(cuda::std::integral_constant<int, 64>{});
    case 80: return fn(cuda::std::integral_constant<int, 80>{});
    default: break;
    }
  }
  return false;
}

// Whether a fused leaf exists for these types, strides, and channel count and
// its working set fits in shared memory. This is a launchability check only.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline bool FusedRadixFeasible(
    const OutType &out, const InType &in, const FilterType &filter,
    index_t num_channels, index_t decimation_factor)
{
  using types = FusedRadixTypes<OutType, InType, FilterType, AccumType>;
  auto fits = [&](auto channels_c) {
    constexpr int channels = decltype(channels_c)::value;
    return FusedRadixFits<channels, OutType, InType, FilterType, AccumType>(
        out, in, filter, decimation_factor);
  };
  if constexpr (types::Small) {
    if (decimation_factor == num_channels && num_channels >= 2 && num_channels <= 6) {
      return VisitFusedRadixChannels<true>(num_channels, fits);
    }
  }
  if constexpr (types::General) {
    if (out.Stride(OutType::Rank() - 1) != 1 || filter.Stride(FilterType::Rank() - 1) != 1) {
      return false;
    }
    if constexpr (is_tensor_view_v<InType>) {
      if (in.Stride(InType::Rank() - 1) != 1) return false;
    }
    return VisitFusedRadixChannels<false>(num_channels, fits);
  }
  return false;
}

// Performance preference for a feasible fused launch. A critical small-channel
// leaf whose FIR tile does not fit in shared memory runs the uncached direct FIR,
// which dispatch reserves for outputs that cuFFT cannot produce.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline bool PreferFusedRadix(const InType &in, const FilterType &filter, index_t num_channels,
                             index_t decimation_factor, DeviceAttrs &attrs)
{
  using types = FusedRadixTypes<OutType, InType, FilterType, AccumType>;
  const index_t input_length = in.Size(InType::Rank() - 1);
  const size_t input_bytes = static_cast<size_t>(TotalSize(in)) *
      sizeof(typename InType::value_type);
  const index_t taps = (filter.Size(FilterType::Rank() - 1) + num_channels - 1) / num_channels;
  return VisitFusedRadixChannels<!types::General>(num_channels, [&](auto channels_c) {
    constexpr int channels = decltype(channels_c)::value;
    if constexpr (channels >= 3 && channels <= 6) {
      if (decimation_factor == channels &&
          !FusedSmallCachedFits<channels, InType, FilterType, AccumType>(taps)) {
        return false;
      }
    }
    return FusedRadixPreferred<channels, typename InType::value_type, AccumType>(
        decimation_factor, input_length, input_bytes, attrs);
  });
}

// Launch the fused leaf for num_channels. The caller checks feasibility.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void RunFusedRadix(
    OutType out, const InType &in, const FilterType &filter,
    index_t num_channels, index_t decimation_factor, cudaStream_t stream, index_t out_elem_offset)
{
  using types = FusedRadixTypes<OutType, InType, FilterType, AccumType>;
  if constexpr (types::Small || types::General) {
    VisitFusedRadixChannels<!types::General>(num_channels, [&](auto channels_c) {
      constexpr int channels = decltype(channels_c)::value;
      if (decimation_factor == channels) {
        FusedRadixImpl<channels, true, OutType, InType, FilterType, AccumType>(
            out, in, filter, decimation_factor, stream, out_elem_offset);
        return true;
      }
      if constexpr (types::General) {
        FusedRadixImpl<channels, false, OutType, InType, FilterType, AccumType>(
            out, in, filter, decimation_factor, stream, out_elem_offset);
        return true;
      }
      return false;
    });
  }
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
  const int gy = static_cast<int>(std::min<index_t>(
      (num_elem_per_channel + THREADS - 1) / THREADS, 65535));
  const dim3 grid(num_batches, gy);

  auto launch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    ChannelizePoly1DUnpackDFT<IsUnitStride, DataType><<<grid, THREADS, 0, stream>>>(inout);
  };
  DispatchUnitStride(launch, inout);
#endif
}

// CUDA backends. Fused computes the FIR and DFT in one kernel. The others run
// a FIR kernel followed by a cuFFT transform (and an unpack for real inputs).
enum class Backend { Fused, Smem, Tiled, Generic };

struct Plan {
  Backend backend = Backend::Generic;
  // Tiled: tile height and filter layout.
  SmemTiledPlan tiled{};
};

// The FIR backends run the DFT with cuFFT, whose half-precision transforms
// require a power-of-two size.
template <typename OutType>
inline bool SeparateFftSupported(index_t num_channels)
{
  if constexpr (is_complex_half_v<typename OutType::value_type>) {
    return (num_channels & (num_channels - 1)) == 0;
  }
  return true;
}

// Dispatch policy:
//  1. A fused leaf whenever one applies and is preferred (PreferFusedRadix), or
//     whenever cuFFT cannot produce the output (SeparateFftSupported).
//  2. Otherwise a FIR kernel followed by cuFFT. Up to SmallChannelLimit
//     channels, the whole-channel Smem kernel (critical) or Generic
//     (oversampled); wider channel counts use the tiled kernel
//     (SelectTiledPlan), except that critical FP64 on low-FP64 GPUs keeps
//     Smem when the last channel tile would be partly idle.
//  3. Generic when no tile fits the default shared-memory limit.
// Without a fused leaf, outputs that cuFFT cannot produce get an unlaunchable
// plan, which ExecutePlan rejects.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline Plan SelectPlan(
    const OutType &out, const InType &in, const FilterType &filter,
    index_t num_channels, index_t decimation_factor, DeviceAttrs &attrs)
{
  Plan plan;
  if (FusedRadixFeasible<OutType, InType, FilterType, AccumType>(
          out, in, filter, num_channels, decimation_factor) &&
      (!SeparateFftSupported<OutType>(num_channels) ||
       PreferFusedRadix<OutType, InType, FilterType, AccumType>(
           in, filter, num_channels, decimation_factor, attrs))) {
    plan.backend = Backend::Fused;
    return plan;
  }
  const bool critical = decimation_factor == num_channels;
  // With low FP64 throughput, the whole-channel Smem kernel also beats a tiled
  // kernel whose last channel tile is partly idle.
  bool smem_preferred = num_channels <= SmallChannelLimit;
  if constexpr (cuda::std::is_same_v<AccumType, double>) {
    smem_preferred = smem_preferred ||
        (num_channels % SmemTiledCtile != 0 && attrs.LowFp64Throughput());
  }
  if (critical && smem_preferred && ShouldUseSmem(out, in, filter)) {
    plan.backend = Backend::Smem;
    return plan;
  }
  if (!critical && num_channels <= SmallChannelLimit) {
    return plan;
  }
  plan.tiled = SelectTiledPlan(out, in, filter, decimation_factor);
  if (SmemTiledFits(plan.tiled, num_channels)) {
    plan.backend = Backend::Tiled;
  }
  return plan;
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline Plan SelectPlan(
    const OutType &out, const InType &in, const FilterType &filter,
    index_t num_channels, index_t decimation_factor)
{
  DeviceAttrs attrs;
  return SelectPlan<OutType, InType, FilterType, AccumType>(
      out, in, filter, num_channels, decimation_factor, attrs);
}

// Whether a plan can be launched for these operands. Selected plans can unless
// no backend can produce the output; this also guards explicitly constructed plans.
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline bool PlanIsLaunchable(
    const Plan &plan, const OutType &out, const InType &in,
    const FilterType &filter, index_t num_channels, index_t decimation_factor)
{
  if (plan.backend != Backend::Fused && !SeparateFftSupported<OutType>(num_channels)) {
    return false;
  }
  switch (plan.backend) {
  case Backend::Fused:
    return FusedRadixFeasible<OutType, InType, FilterType, AccumType>(
        out, in, filter, num_channels, decimation_factor);
  case Backend::Smem:
    return decimation_factor == num_channels && ShouldUseSmem(out, in, filter);
  case Backend::Tiled: {
    const int nout = plan.tiled.nout;
    const bool nout_ok = nout == SmemTiledNout ||
        (nout == SmemTiledMaxDecNout && decimation_factor == num_channels);
    const auto sized = SmemTiledPlanWithLayout(
        out, in, filter, decimation_factor, nout, plan.tiled.filter_layout);
    return nout_ok && sized.bytes == plan.tiled.bytes &&
        SmemTiledFits(sized, num_channels) &&
        (plan.tiled.filter_layout != SmemTiledFilterLayout::Rotated ||
         SmemTiledRotatedLayoutSupported(num_channels, decimation_factor));
  }
  case Backend::Generic:
    return true;
  }
  return false;
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void ExecutePlan(
    const Plan &plan, OutType out, const InType &in, const FilterType &f,
    index_t num_channels, index_t decimation_factor, cudaStream_t stream,
    index_t out_elem_offset, DeviceAttrs &attrs)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  using output_t = typename OutType::value_type;
  constexpr int OUT_RANK = OutType::Rank();

  if (!PlanIsLaunchable<OutType, InType, FilterType, AccumType>(
          plan, out, in, f, num_channels, decimation_factor)) {
    MATX_THROW(matxInvalidParameter, SeparateFftSupported<OutType>(num_channels)
        ? "channelize_poly: backend plan is not supported for these operands"
        : "channelize_poly: half-precision outputs need a power-of-two channel count "
          "unless a fused kernel applies (critical sampling with 3, 5, or 6 channels)");
  }

  if (plan.backend == Backend::Fused) {
    RunFusedRadix<OutType, InType, FilterType, AccumType>(
        out, in, f, num_channels, decimation_factor, stream, out_elem_offset);
    return;
  }

  auto run_filter_backend = [&](auto filtered_output) {
    using filtered_output_t = decltype(filtered_output);
    switch (plan.backend) {
    case Backend::Smem:
      Smem<filtered_output_t, InType, FilterType, AccumType>(
          filtered_output, in, f, stream, out_elem_offset, attrs);
      break;
    case Backend::Tiled:
      SmemTiled<filtered_output_t, InType, FilterType, AccumType>(
          filtered_output, in, f, decimation_factor, plan.tiled, stream, out_elem_offset, attrs);
      break;
    default:
      Generic<filtered_output_t, InType, FilterType, AccumType>(
          filtered_output, in, f, decimation_factor, stream, out_elem_offset);
      break;
    }
  };

  // If neither the input nor the filter is complex, then the filtered samples will be real-valued
  // and we will use an R2C transform. Otherwise, we will use a C2C transform.
  if constexpr (!is_complex_v<input_t> && !is_complex_v<filter_t>) {
    index_t start_dims[OUT_RANK], stop_dims[OUT_RANK];
    std::fill_n(start_dims, OUT_RANK, 0);
    std::fill_n(stop_dims, OUT_RANK, matxEnd);

    // The first kernel below needs a buffer of type input_t (known to be real in this constexpr
    // branch) into which we store filtered data prior to the real-to-complex FFT. If the output
    // buffer is contiguous, then we use an aliased tensor view of type input_t for that buffer
    // where the last dimension is twice as large (because input_t is real and output_t is complex).
    // We then use a slice to maintain the expected dimensions. If the output buffer is not
    // contiguous, then we async allocate a temporary buffer. There is one caveat with this
    // allocate: the batched fft implementation currently requires that all input pointers must be
    // aligned to the corresponding complex type, which cannot be guaranteed to always be true for a
    // real-valued tensor. This was not an issue for the reused output buffer because the output
    // tensor is complex-valued, so we always have an even stride from one batch to the next. As a
    // temporary workaround for the FFT alignment issue, we add one channel in the odd-channel case
    // and use a slice to create a tensor view of only [0, num_channels-1]. This guarantees that we
    // always stride by an even number of elements from one batch to the next while exposing a
    // tensor view of appropriate dimensions.
    using post_filter_t = typename inner_op_type_t<output_t>::type;
    auto fft_in_slice = [&out, &start_dims, &stop_dims, num_channels, stream]() -> auto {
        auto fft_in_shape = out.Shape();
        if (out.IsContiguous()) {
          fft_in_shape[OUT_RANK-1] *= 2;
          auto fft_in = make_tensor<post_filter_t>(
              reinterpret_cast<post_filter_t*>(out.Data()), fft_in_shape);
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

    run_filter_backend(fft_in_slice);
    stop_dims[OUT_RANK-1] = (num_channels/2) + 1;
    auto out_packed = slice<OUT_RANK>(out, start_dims, stop_dims);
    (out_packed = fft(fft_in_slice, num_channels)).run(stream);
    UnpackDFT(out, stream);
  } else {
    run_filter_backend(out);
    // Specify FORWARD here to prevent any normalization after the ifft. We do not
    // want any extra scaling on the output values.
    (out = ifft(out, num_channels, FFTNorm::FORWARD)).run(stream);
  }
}

template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void ExecutePlan(
    const Plan &plan, OutType out, const InType &in, const FilterType &f,
    index_t num_channels, index_t decimation_factor, cudaStream_t stream,
    index_t out_elem_offset = 0)
{
  DeviceAttrs attrs;
  ExecutePlan<OutType, InType, FilterType, AccumType>(
      plan, out, in, f, num_channels, decimation_factor, stream, out_elem_offset, attrs);
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
 * @param out_elem_offset Global index of the first output element (time step)
 * to compute. The output tensor's second-to-last dimension is sized to the
 * requested window, and the elements written are those with global time
 * indices [out_elem_offset, out_elem_offset + window). The default of 0 with a
 * full-length output computes the entire output; a windowed call computes only
 * that range (used by the streaming channelizer).
 */
template <typename OutType, typename InType, typename FilterType, typename AccumType>
inline void channelize_poly_impl(OutType out, const InType &in, const FilterType &f,
                   index_t num_channels, index_t decimation_factor, cudaStream_t stream = 0,
                   index_t out_elem_offset = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using OutputOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<OutType>>;
  using InputOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<InType>>;
  using FilterOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<FilterType>>;

  detail::cpoly::ValidateChannelizePolyArgs<AccumType>(
      out, in, f, num_channels, decimation_factor, out_elem_offset);
  // An empty signal, batch dimension, or output window leaves nothing to compute.
  if (TotalSize(out) == 0) {
    return;
  }

  detail::cpoly::DeviceAttrs attrs;
  const auto plan = detail::cpoly::SelectPlan<
      OutputOp, InputOp, FilterOp, AccumType>(
          out, in, f, num_channels, decimation_factor, attrs);
  detail::cpoly::ExecutePlan<OutputOp, InputOp, FilterOp, AccumType>(
      plan, out, in, f, num_channels, decimation_factor, stream, out_elem_offset, attrs);
}

/**
 * @brief Host implementation of the 1D polyphase channelizer.
 *
 * This is a feature-parity implementation for CPU executors. It directly
 * computes the per-branch FIR values and then applies the unnormalized,
 * positive-sign DFT used by the CUDA channelizer.
 *
 * @tparam OutType Type of output
 * @tparam InType Type of input
 * @tparam FilterType Type of filter
 * @tparam AccumType Type of accumulator. This type should always be real, but
 * it will be promoted to complex when necessary.
 * @tparam MODE Host executor threading mode
 * @param out Output tensor
 * @param in Input operator
 * @param f Filter operator
 * @param num_channels Number of channels in which to separate the signal
 * @param decimation_factor Factor by which to downsample the input signal into
 * the channels
 * @param exec Host executor on which to run
 * @param out_elem_offset Global index of the first output element (time step)
 * to compute, with the same window semantics as the CUDA overload: the output
 * tensor's second-to-last dimension is sized to the requested window, and the
 * elements written are those with global time indices
 * [out_elem_offset, out_elem_offset + window). The default of 0 with a
 * full-length output computes the entire output.
 */
template <typename OutType, typename InType, typename FilterType, typename AccumType, ThreadsMode MODE>
inline void channelize_poly_impl(OutType out, const InType &in, const FilterType &f,
                   index_t num_channels, index_t decimation_factor,
                   [[maybe_unused]] const HostExecutor<MODE> &exec,
                   index_t out_elem_offset = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using OutputOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<OutType>>;
  using InputOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<InType>>;
  using FilterOp = cuda::std::remove_cv_t<cuda::std::remove_reference_t<FilterType>>;
  using input_t = typename InputOp::value_type;
  using filter_t = typename FilterOp::value_type;
  using filtering_accum_t = cuda::std::conditional_t<
      is_complex_v<input_t> || is_complex_v<filter_t>,
      typename detail::scalar_to_complex<AccumType>::ctype,
      AccumType>;
  using complex_accum_t = typename detail::scalar_to_complex<AccumType>::ctype;

  constexpr int IN_RANK = InputOp::Rank();
  constexpr int OUT_RANK = OutputOp::Rank();
  detail::cpoly::ValidateChannelizePolyArgs<AccumType>(
      out, in, f, num_channels, decimation_factor, out_elem_offset);

  const index_t input_len = in.Size(IN_RANK - 1);
  const index_t out_rows = out.Size(OUT_RANK - 2);

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
    // t is the local write row; tg is the global output element (time) index
    // that drives the input footprint and polyphase phase (tg == t when not
    // windowed).
    const index_t tg = t + out_elem_offset;
    auto [input_b, output_b, filter_acc] = detail::cpoly::MakeAccessors<false>(out, in, f, batch);
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
        sample_idx = s + tg * num_channels;
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

        if (available_taps > h_skip && (tg + 1) > h_skip) {
          niter = std::min(available_taps - h_skip, tg + 1 - h_skip);
          h_ind = branch + h_skip * num_channels;
        }
      } else {
        const index_t r_remapped = (branch + num_channels - decimation_factor) % num_channels;
        const index_t s = num_channels - 1 - r_remapped;
        const index_t last_arrived = tg * decimation_factor + decimation_factor - 1;
        if (last_arrived >= s) {
          const index_t A = last_arrived - s;
          sample_idx = last_arrived - (A % num_channels);
          const index_t causal_count = A / num_channels + 1;
          const index_t phase = (branch + tg * decimation_factor) % num_channels;
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
        const input_t in_val = input_b(sample_idx);
        const filter_t h_val = filter_acc(h_ind);
        detail::channelize_cmac(
            accum, detail::channelize_cast_operand<filtering_accum_t>(h_val),
            detail::channelize_cast_operand<filtering_accum_t>(in_val));
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
      output_b(t, channel) = static_cast<typename OutputOp::value_type>(dft);
    }
  };

  const index_t total_outputs = batch_count * out_rows;
#ifdef MATX_EN_OMP
  if (exec.GetNumThreads() > 1) {
    #pragma omp parallel for num_threads(exec.GetNumThreads())
    for (index_t i = 0; i < total_outputs; i++) {
      compute_output(i / out_rows, i % out_rows);
    }
  } else
#endif
  {
    for (index_t i = 0; i < total_outputs; i++) {
      compute_output(i / out_rows, i % out_rows);
    }
  }
}
} // end namespace matx
