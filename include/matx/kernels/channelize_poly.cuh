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

#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/kernels/tensor_accessor.h"
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/tuple>

namespace matx {
namespace detail {

// Packed even/odd values used only by the specialized M=2 leaves.
template <typename T>
struct alignas(sizeof(T) < 16 ? 2 * sizeof(T) : 16) ChannelizePolyM2Pair {
    T even;
    T odd;
};

// detail constants that require both host and device visibility at compile
// time. Scoped to matx::detail::cpoly so the transform's helpers can sit in
// the same namespace without colliding with other transforms' internals.
namespace cpoly {
    // Number of output elements generated per thread
    constexpr index_t ElemsPerThread = 1;

    // Largest dynamic shared memory allocation a kernel can launch with without
    // opting in through cudaFuncSetAttribute.
    constexpr size_t MaxDefaultDynamicSmemBytes = 48 * 1024;

    // Number of independent time lanes in a maximally decimated tiled CTA. Each lane computes
    // several output rows so its channel's filter tap can be reused across those rows.
    constexpr int SmemTiledMaxDecYThreads = 4;

    // Maximum number of filter rotations per channel for the SmemTiled kernel.
    // This is used to determine if the filter can be stored in shared memory.
    // The number of rotations can exceed this value, but the filter will be
    // read from global memory rather than cached in shared memory.
    constexpr int SmemTiledMaxRotations = 32;

    __MATX_HOST__ __MATX_DEVICE__ constexpr index_t SmemTiledInputHeight(
        index_t taps, index_t channels, index_t decimation, int nout)
    {
        // Near-critical oversampling can need one older branch row while
        // the next output group has already loaded a newer row into the ring.
        const bool extra_row = decimation < channels &&
            static_cast<int64_t>(nout - 1) * decimation >
                static_cast<int64_t>(nout - 2) * channels;
        return taps + nout - 1 + extra_row;
    }

    constexpr int FusedRadixThreads = 256;

    // Zero-padded load of input(base + offset). Requires offset >= 0. Unsigned
    // arithmetic wraps modulo 2^N without overflow; for any representable base,
    // a negative base + offset maps to at least 2^(N-1) > input_len, and a sum
    // cannot wrap past 2^N, so one compare checks both bounds exactly.
    template <typename Input, typename IdxT>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto LoadRelativeInput(
        const Input &input, IdxT base, IdxT input_len, int32_t offset)
    {
        using input_t = cuda::std::remove_cvref_t<decltype(input(base))>;
        using unsigned_index_t = cuda::std::make_unsigned_t<IdxT>;
        const auto index = static_cast<unsigned_index_t>(base) +
            static_cast<unsigned_index_t>(offset);
        return index < static_cast<unsigned_index_t>(input_len)
            ? input(static_cast<IdxT>(index)) : input_t{};
    }

    // Small channel counts are most efficient when a thread owns a complete output row. Two rows
    // per thread provide useful FIR ILP for narrow FP32 leaves. One row bounds register and
    // shared-memory use for M>=4 and for FP64 leaves above M=2.
    template <int NUM_CHAN, typename AccumType>
    __MATX_HOST__ __MATX_DEVICE__ constexpr int FusedSmallOutputsPerThread()
    {
        static_assert(NUM_CHAN >= 2 && NUM_CHAN <= 6);
        if constexpr (NUM_CHAN >= 4 || (cuda::std::is_same_v<AccumType, double> && NUM_CHAN >= 3)) {
            return 1;
        } else {
            return 2;
        }
    }

    // Staging a complete overlapping tile is not amortized by very short filters. Keep that case
    // inside the same kernel instantiation, but read the few FIR values directly before applying
    // the fixed butterfly. Real input can use the direct leaf a little longer because its
    // global-memory footprint is half that of complex input.
    template <int NUM_CHAN, typename InputType, typename AccumType>
    __MATX_HOST__ __MATX_DEVICE__ constexpr int FusedSmallDirectMaxTaps()
    {
        if constexpr (cuda::std::is_same_v<InputType, AccumType> && NUM_CHAN >= 5) {
            return 16;
        } else {
            return 4;
        }
    }

    __MATX_HOST__ __MATX_DEVICE__ constexpr int FusedRadixOddFactor(int n)
    {
        while ((n & 1) == 0) n /= 2;
        return n;
    }

    __MATX_HOST__ __MATX_DEVICE__ constexpr int FusedRadixRowStride(int n)
    {
        // Pack oversampled M=3..6 rows into subwarps. Critical small-M leaves
        // use their separate row-owned mapping.
        int stride = n >= 3 && n <= 6 ? 1 : 32;
        while (stride < n) stride *= 2;
        return stride;
    }

    // Reverse the lowest log2(N) bits for a power-of-two FFT size N.
    template <int N>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ int32_t BitReverse(int32_t index)
    {
        static_assert(N > 0 && (N & (N - 1)) == 0);
        int32_t reversed = 0;
        uint32_t remaining = static_cast<uint32_t>(index);
        // BREV/CLZ could replace this loop, but measured slightly slower.
        MATX_LOOP_UNROLL
        for (int bit = 1; bit < N; bit *= 2) {
            reversed = 2 * reversed + static_cast<int32_t>(remaining & 1);
            remaining /= 2;
        }
        return reversed;
    }

    template <int NUM_CHAN, typename AccumType,
              bool MaximallyDecimated = false, typename InputType = AccumType>
    struct FusedRadixConfig {
        static_assert(NUM_CHAN >= 2);
        static constexpr int Radix1 = FusedRadixOddFactor(NUM_CHAN);
        static constexpr int Radix2 = NUM_CHAN / Radix1;
        static constexpr bool PackedPow2 = Radix1 == 1 && NUM_CHAN >= 8 && NUM_CHAN < 32;
        // Fill subwarps while retaining the existing tile height and footprint.
        static constexpr int PackedRows = cuda::std::is_same_v<AccumType, double> ? 16 : 32;
        // Smaller row-owned CTAs improve coverage. Real M=5/6 keeps 128 threads
        // to amortize FIR staging without sacrificing short-signal coverage.
        static constexpr int SmallThreads =
            (NUM_CHAN == 2 && cuda::std::is_same_v<AccumType, double>) ||
            (NUM_CHAN >= 3 && NUM_CHAN <= 4) ||
            (is_complex_v<InputType> && (NUM_CHAN >= 3 || sizeof(InputType) > 8))
                ? 64 : NUM_CHAN >= 5 ? 128 : FusedRadixThreads;
        static constexpr int Threads =
            MaximallyDecimated && NUM_CHAN <= 6 ? SmallThreads :
            PackedPow2 ? cuda::std::min(FusedRadixThreads, PackedRows * NUM_CHAN) :
            FusedRadixThreads;
        static constexpr int RowStride = PackedPow2 ? NUM_CHAN : FusedRadixRowStride(NUM_CHAN);
        // Packed FIR rows retain their low-overhead tiny DFT.
        static constexpr bool WarpFft = (RowStride >= 32 || PackedPow2) && Radix2 <= 32;
        // Large register butterflies need enough resident warps to hide latency.
        static constexpr int MinBlocksPerSm =
            WarpFft && 2 * Radix1 * sizeof(AccumType) >= 64 ? 1024 / Threads : 0;
        static constexpr int FirGroups = Threads / RowStride;
        // One output per thread bounds the packed small-M tile's footprint.
        // Wide double FIR tiles retain smaller groups to leave room for taps.
        static constexpr int FirOutputsPerThread =
            NUM_CHAN >= 3 && NUM_CHAN <= 6 ? 1 :
            PackedPow2 ? PackedRows * NUM_CHAN / Threads :
            cuda::std::is_same_v<AccumType, double>
                ? (!WarpFft ? 1 : Radix1 == 1 ? 2 : cuda::std::min(FirGroups, 4)) : 4;
        static constexpr int NRows = NUM_CHAN == 2
            ? 1 : FirGroups * FirOutputsPerThread;
        static constexpr int SmallOutputsPerThread = [] {
            if constexpr (NUM_CHAN <= 6) {
                return FusedSmallOutputsPerThread<NUM_CHAN, AccumType>();
            } else {
                return 0;
            }
        }();
    };

    // Shared by launch sizing and device access. Element counts use native
    // input/filter types, including when M=2 accesses them as packed pairs.
    template <int NUM_CHAN, bool MaximallyDecimated, typename InputType,
              typename FilterType, typename AccumType>
    struct FusedRadixSmemLayout {
        index_t filter_elements;
        index_t input_elements;
        size_t input_offset;
        size_t work_offset = 0;
        size_t stage1_offset = 0;
        size_t twiddle_cross_offset = 0;
        size_t twiddle_radix2_offset = 0;
        size_t bytes;

        // Preserve the caller's index width: device tiles use 32-bit counts,
        // while host eligibility checks may size filters too large to launch.
        template <typename IdxT>
        __MATX_HOST__ __MATX_DEVICE__ constexpr FusedRadixSmemLayout(
            IdxT taps_per_channel, index_t decimation_factor)
        {
            using config = FusedRadixConfig<NUM_CHAN, AccumType, MaximallyDecimated, InputType>;
            using complex_t = typename scalar_to_complex<AccumType>::ctype;
            constexpr size_t input_alignment =
                NUM_CHAN == 2 && MaximallyDecimated
                    ? alignof(ChannelizePolyM2Pair<InputType>)
                    : alignof(InputType);
            filter_elements = taps_per_channel * NUM_CHAN;
            input_offset = MATX_ROUND_UP(
                static_cast<size_t>(taps_per_channel) * NUM_CHAN *
                    sizeof(FilterType),
                input_alignment);
            if constexpr (NUM_CHAN <= 6 && MaximallyDecimated) {
                constexpr int block_rows = config::Threads * config::SmallOutputsPerThread;
                input_elements = (taps_per_channel + block_rows - 1) * NUM_CHAN;
            } else if constexpr (NUM_CHAN == 2) {
                input_elements = 2 * taps_per_channel +
                    config::Threads * config::SmallOutputsPerThread;
            } else if constexpr (MaximallyDecimated) {
                input_elements = (taps_per_channel + config::NRows - 1) * NUM_CHAN;
            } else {
                input_elements = taps_per_channel * NUM_CHAN +
                    (config::NRows - 1) * static_cast<IdxT>(decimation_factor) + 1;
            }
            bytes = input_offset + static_cast<size_t>(input_elements) * sizeof(InputType);
            if constexpr (NUM_CHAN > 2 && !(NUM_CHAN <= 6 && MaximallyDecimated)) {
                constexpr size_t work_bytes = config::NRows * NUM_CHAN * sizeof(complex_t);
                work_offset = MATX_ROUND_UP(bytes, alignof(complex_t));
                bytes = work_offset + work_bytes;
                if constexpr (config::Radix1 > 1) {
                    stage1_offset = bytes;
                    twiddle_cross_offset = stage1_offset + (config::WarpFft ? 0 : work_bytes);
                    bytes = twiddle_cross_offset + NUM_CHAN * sizeof(complex_t);
                }
                twiddle_radix2_offset = bytes;
                bytes = twiddle_radix2_offset +
                    (config::Radix2 + config::Radix2 / 2) * sizeof(complex_t);
            }
        }
    };

    // Whether the cached small-channel leaf's FIR tile fits the default
    // shared-memory limit.
    template <int NUM_CHAN, typename InType, typename FilterType, typename AccumType>
    __MATX_HOST__ __MATX_DEVICE__ constexpr bool FusedSmallCachedFits(index_t taps_per_channel)
    {
        const FusedRadixSmemLayout<NUM_CHAN, true, typename InType::value_type,
            typename FilterType::value_type, AccumType> layout(taps_per_channel, NUM_CHAN);
        return layout.bytes <= MaxDefaultDynamicSmemBytes;
    }

    // Keep host allocation and device leaf selection in sync. The direct leaf is
    // faster for short filters and is the fallback when the cached tile does not fit.
    template <int NUM_CHAN, typename OutType, typename InType,
              typename FilterType, typename AccumType>
    __MATX_HOST__ __MATX_DEVICE__ constexpr bool FusedSmallUseDirect(
        index_t taps_per_channel, index_t output_rows)
    {
        using input_t = typename InType::value_type;
        if constexpr (is_tensor_view_v<InType>) {
            bool use_direct = taps_per_channel <=
                FusedSmallDirectMaxTaps<NUM_CHAN, input_t, AccumType>();
            if constexpr (!cuda::std::is_same_v<input_t, AccumType> && NUM_CHAN >= 5) {
                // For very long outputs, caching saves enough global reads
                // to outweigh the direct leaf's lower setup cost.
                use_direct &= output_rows <= (index_t{1} << 20);
            }
            if (use_direct) return true;
        }
        return !FusedSmallCachedFits<NUM_CHAN, InType, FilterType, AccumType>(taps_per_channel);
    }

    // Build and batch-bind the FIR accessors in their common setup order.
    template <bool IsUnitStride, typename OutType, typename InType, typename FilterType>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto MakeAccessors(
        const OutType &output, const InType &input, const FilterType &filter, index_t batch)
    {
        TensorAccessor<InType, IsUnitStride> input_acc(input);
        TensorAccessor<OutType, IsUnitStride> output_acc(output);
        TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);
        const auto in_batch_idx = BlockToIdx(input, batch, 1);
        const auto out_batch_idx = BlockToIdx(output, batch, 2);
        auto input_b = bind_first_n<InType::Rank() - 1>(input_acc, in_batch_idx);
        auto output_b = bind_first_n<OutType::Rank() - 2>(output_acc, out_batch_idx);
        return cuda::std::make_tuple(input_b, output_b, filter_acc);
    }
} // namespace cpoly

template <typename AccumT, typename ValueT>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto channelize_cast_operand(ValueT v)
{
    if constexpr (cuda::std::is_same_v<AccumT, ValueT>) {
        return v;
    } else if constexpr (is_complex_v<ValueT>) {
        // Component-wise conversion also supports mixed complex-half types.
        using scalar_t = typename inner_op_type_t<AccumT>::type;
        return AccumT{static_cast<scalar_t>(v.real()), static_cast<scalar_t>(v.imag())};
    } else if constexpr (is_complex_v<AccumT>) {
        // Preserve a real operand so channelize_cmac can use its cheaper
        // real-by-complex specialization.
        using accum_scalar_t = typename inner_op_type_t<AccumT>::type;
        return static_cast<accum_scalar_t>(v);
    } else {
        return static_cast<AccumT>(v);
    }
}
// Fused complex multiply-accumulate. Decomposes the operation into scalar
// FMA instructions so the compiler emits 4 FFMA per complex tap instead of
// ~8 mixed FMUL/FADD/FSUB. Falls back to the default operator* + operator+=
// for real or mixed-precision types.
template <typename AccumT, typename FilterValT, typename InputValT>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void channelize_cmac(
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
        // Real filter * complex input
        auto a_re = accum.real(), a_im = accum.imag();
        a_re = hv * iv.real() + a_re;
        a_im = hv * iv.imag() + a_im;
        accum = {a_re, a_im};
    } else if constexpr (is_complex_v<AccumT> && is_complex_v<FilterValT> && !is_complex_v<InputValT>) {
        // Complex filter * real input
        auto a_re = accum.real(), a_im = accum.imag();
        a_re = hv.real() * iv + a_re;
        a_im = hv.imag() * iv + a_im;
        accum = {a_re, a_im};
    } else {
        accum += hv * iv;
    }
}

} // namespace detail

#ifdef __CUDACC__

// out_elem_offset shifts the global per-channel output element (time) index
// used for the input footprint and polyphase phase, while the write row stays
// local, so this kernel can emit an arbitrary window
// [out_elem_offset, out_elem_offset + output.Size(OutElemRank)] of the full
// output-element grid. out_elem_offset is 0 for a normal full-grid call, but
// can be non-zero for a streaming channelizer call.
template <int THREADS, bool MaximallyDecimated, bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename AccumType>
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D(
    OutType output, InType input, FilterType filter, index_t decimation_factor,
    uint32_t smem_filter_bytes, index_t out_elem_offset, int elem_block_offset)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;
    static_assert(! is_complex_v<AccumType>,
      "channelize_poly: accumulator type must be real; it will be treated as complex when necessary");
    // If the output is complex, then then accumulator is complex. Otherwise, the accumulator is real.
    using accum_t = cuda::std::conditional_t<is_complex_v<output_t>, typename detail::scalar_to_complex<AccumType>::ctype, AccumType>;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int ChannelRank = OutRank-1;
    constexpr int OutElemRank = OutRank-2;

    const index_t input_len = input.Size(InRank-1);
    const index_t output_len_per_channel = output.Size(OutElemRank);
    const index_t num_channels = output.Size(ChannelRank);
    const index_t filter_full_len = filter.Size(0);
    const index_t filter_phase_len = (filter_full_len + num_channels - 1) / num_channels;

    // Channels vary fastest across CTAs so the CTAs that read the same input
    // span run together and share it through L2. Time blocks use grid.y; the
    // launch splits grids taller than the grid.y limit using elem_block_offset.
    const int channel = static_cast<int>(blockIdx.x);
    const int elem_block = static_cast<int>(blockIdx.y) + elem_block_offset;
    const int tid = threadIdx.x;

    constexpr index_t ELEMS_PER_BLOCK = detail::cpoly::ElemsPerThread * THREADS;
    const index_t first_out_elem = elem_block * detail::cpoly::ElemsPerThread * THREADS;
    const index_t last_out_elem = cuda::std::min(
        output_len_per_channel - 1, first_out_elem + ELEMS_PER_BLOCK - 1);

    // Bind batch dimensions; output accesses retain (time, channel) indices.
    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    if constexpr (MaximallyDecimated) {
        // Maximally decimated (D == M) path: filter phase is fixed per channel
        // Dynamic shared memory holds one phase's taps when the dispatch provides
        // enough smem. When smem_bytes == 0 the filter is read from global memory.
        // When D == M: phase = channel (fixed), A % M = channel,
        // indims = s + t*M, causal_count = t + 1, and last_arrived >= s
        // is always true for t >= 0.
        const index_t s = num_channels - 1 - channel;

        extern __shared__ __align__(16) uint8_t smem_filter_raw[];
        filter_t *smem_filter = reinterpret_cast<filter_t *>(smem_filter_raw);
        const bool use_smem_filter = (smem_filter_bytes >= sizeof(filter_t) * filter_phase_len);
        if (use_smem_filter) {
            for (index_t t = tid; t < filter_phase_len-1; t += THREADS) {
                const index_t h_ind = channel + t * num_channels;
                smem_filter[t] = filter_acc(h_ind);
            }
            if (tid == THREADS-1) {
                const index_t h_ind = channel + (filter_phase_len-1) * num_channels;
                smem_filter[filter_phase_len-1] = (h_ind < filter_full_len) ?
                    filter_acc(h_ind) : static_cast<filter_t>(0);
            }

            __syncthreads();
        }

        if (use_smem_filter) {
            for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
                const index_t g = t + out_elem_offset; // global output element (time) index
                accum_t accum {};
                index_t sample_idx = s + g * num_channels;
                index_t h_skip = 0;
                if (sample_idx >= input_len) {
                    h_skip = 1;
                    sample_idx -= num_channels;
                }
                const filter_t *h = smem_filter + h_skip;
                int niter = static_cast<int>(cuda::std::min(filter_phase_len - h_skip, g + 1 - h_skip));
                for (int i = 0; i < niter; i++) {
                    const input_t in_val = input_b(sample_idx);
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_operand<accum_t>(*h),
                            detail::channelize_cast_operand<accum_t>(in_val));
                    sample_idx -= num_channels;
                    h++;
                }
                output_b(t, channel) = static_cast<output_t>(accum);
            }
        } else {
            index_t available_taps = filter_phase_len;
            {
                const bool h_is_padded = ((filter_phase_len-1) * num_channels + channel) >= filter_full_len;
                if (h_is_padded) {
                    available_taps--;
                }
            }

            for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
                const index_t g = t + out_elem_offset; // global output element (time) index
                accum_t accum {};
                index_t sample_idx = s + g * num_channels;
                index_t h_skip = 0;
                if (sample_idx >= input_len) {
                    h_skip = 1;
                    sample_idx -= num_channels;
                }
                index_t h_ind = channel + h_skip * num_channels;
                index_t niter = cuda::std::min(available_taps - h_skip, g + 1 - h_skip);
                for (index_t i = 0; i < niter; i++) {
                    const input_t in_val = input_b(sample_idx);
                    const filter_t h_val = filter_acc(h_ind);
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_operand<accum_t>(h_val),
                            detail::channelize_cast_operand<accum_t>(in_val));
                    h_ind += num_channels;
                    sample_idx -= num_channels;
                }
                output_b(t, channel) = static_cast<output_t>(accum);
            }
        }
    } else {
        // Oversampled (D < M) path: phase rotates per output step
        // No shared memory. Reads filter from global/L2.
        // Branch remap for Harris convention: r_remapped changes the input
        // sample mapping so newest D samples land in branches D-1..0.
        // Phase uses the original logical channel index (not remapped).
        const index_t r_remapped = (channel + num_channels - decimation_factor) % num_channels;
        const index_t s = num_channels - 1 - r_remapped;
        for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
            const index_t g = t + out_elem_offset; // global output element (time) index
            const index_t last_arrived = g * decimation_factor + decimation_factor - 1;
            index_t niter = 0;
            const index_t phase = (channel + g * decimation_factor) % num_channels;
            index_t h_ind { phase };
            index_t sample_idx = 0;
            accum_t accum {};
            if (last_arrived >= s) {
                const index_t A = last_arrived - s;
                sample_idx = last_arrived - (A % num_channels);
                const index_t causal_count = A / num_channels + 1;
                index_t h_skip = 0;
                if (sample_idx >= input_len) {
                    h_skip = 1;
                    sample_idx -= num_channels;
                }
                h_ind = phase + h_skip * num_channels;
                index_t available_taps = filter_phase_len;
                {
                    const bool h_is_padded = ((filter_phase_len-1) * num_channels + phase) >= filter_full_len;
                    if (h_is_padded) {
                        available_taps--;
                    }
                }
                niter = cuda::std::min(available_taps - h_skip, causal_count - h_skip);
            }
            for (index_t i = 0; i < niter; i++) {
                const input_t in_val = input_b(sample_idx);
                const filter_t h_val = filter_acc(h_ind);
                detail::channelize_cmac(accum,
                        detail::channelize_cast_operand<accum_t>(h_val),
                        detail::channelize_cast_operand<accum_t>(in_val));
                h_ind += num_channels;
                sample_idx -= num_channels;
            }
            output_b(t, channel) = static_cast<output_t>(accum);
        }
    }
}

// Tiled shared-memory polyphase channelizer kernel.
//
// Tiles across channels so that only CTILE channels are processed per block,
// removing the M<=256 constraint of ChannelizePoly1D_Smem while staging
// input samples in shared memory. Supports both maximally decimated (D == M)
// and oversampled (D < M) cases.
//
// Template parameters:
//   FilterInSmem: when true, filter taps are cached in shared memory;
//                 when false, filter taps are read from global/L2.
//
// Block: dim3(CTILE, MaximallyDecimated ? SmemTiledMaxDecYThreads : NOUT)
// Grid:  dim3(time_blocks, channel_tiles, batches)
//
// Shared memory layout (FilterInSmem = true):
//   smem_filter: [P][CTILE] for D==M, or [CTILE][K][P] for D<M
//   smem_input:  [height][CTILE] circular buffer; see SmemTiledInputHeight
//
// Shared memory layout (FilterInSmem = false):
//   smem_input:  [height][CTILE] circular buffer only
//
// Each column of smem_input is the circular buffer for one branch of the
// commutator. Row r, column cx stores input[s(c) + r*M] where c = tile_base+cx
// and s(c) = M-1-c.
template <int CTILE, int NOUT, bool MaximallyDecimated, bool FilterInSmem, bool FilterFullLayout,
          bool IsUnitStride, typename IdxT,
          typename OutType, typename InType, typename FilterType, typename AccumType>
__launch_bounds__(CTILE * (MaximallyDecimated ? detail::cpoly::SmemTiledMaxDecYThreads : NOUT))
// See ChannelizePoly1D for the out_elem_offset output-element window semantics.
// Every local output element is shifted to its global index before it drives the
// input footprint / phase / circular-buffer block index (via max_bidx and the
// newest_raw/last_arrived/phase computations); the write row stays local.
__global__ void ChannelizePoly1D_SmemTiled(
    OutType output, InType input, FilterType filter,
    IdxT elems_per_channel_per_cta, IdxT decimation_factor, int32_t num_phases_per_channel,
    IdxT out_elem_offset)
{
    using output_t = typename OutType::value_type;
    using input_t  = typename InType::value_type;
    using filter_t = typename FilterType::value_type;
    static_assert(!is_complex_v<AccumType>,
        "channelize_poly: accumulator type must be real; it will be treated as complex when necessary");
    using accum_t = cuda::std::conditional_t<is_complex_v<output_t>,
        typename detail::scalar_to_complex<AccumType>::ctype, AccumType>;

    extern __shared__ __align__(16) uint8_t smem_raw[];

    constexpr int InRank  = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int ChannelRank = OutRank - 1;
    constexpr int OutElemRank = OutRank - 2;

    const IdxT input_len = static_cast<IdxT>(input.Size(InRank - 1));
    const IdxT output_len_per_channel = static_cast<IdxT>(output.Size(OutElemRank));
    const int32_t M = static_cast<int32_t>(output.Size(ChannelRank));
    const int32_t filter_full_len = static_cast<int32_t>(filter.Size(0));
    const int32_t P = static_cast<int32_t>((filter_full_len + M - 1) / M);
    const int32_t K = num_phases_per_channel;

    const int32_t cx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int32_t tid = ty * CTILE + cx;
    assert(blockDim.x * blockDim.y == CTILE *
        (MaximallyDecimated ? detail::cpoly::SmemTiledMaxDecYThreads : NOUT) && blockDim.z == 1);
    const int32_t nthreads = CTILE *
        (MaximallyDecimated ? detail::cpoly::SmemTiledMaxDecYThreads : NOUT);
    const int32_t tile_base = static_cast<int32_t>(blockIdx.y) * CTILE;
    const int32_t c = tile_base + cx;
    const bool active = (c < M);

    // Branch remap offset for Harris convention (oversampled only; 0 for D == M).
    const int32_t L = MaximallyDecimated ? 0 : (M - static_cast<int32_t>(decimation_factor));

    const int32_t filter_stride = K * P; // per-channel filter block size
    const int32_t height = MaximallyDecimated ? P + NOUT - 1 :
        static_cast<int32_t>(detail::cpoly::SmemTiledInputHeight(P, M, decimation_factor, NOUT));

    filter_t *smem_filter_base = nullptr;
    input_t  *smem_input = nullptr;
    if constexpr (FilterInSmem) {
        smem_filter_base = reinterpret_cast<filter_t *>(smem_raw);
        // Filter smem slot count depends on chosen layout:
        //   Full:    P * M unique taps
        //   Rotated: per-channel redundant (CTILE * P for D==M, CTILE * K * P for D<M)
        const int32_t filter_elems = FilterFullLayout
            ? (P * M)
            : (MaximallyDecimated ? (P * CTILE) : (CTILE * filter_stride));
        size_t input_byte_offset = sizeof(filter_t) * filter_elems;
        if (input_byte_offset % sizeof(input_t)) {
            input_byte_offset += sizeof(input_t) - input_byte_offset % sizeof(input_t);
        }
        smem_input = reinterpret_cast<input_t *>(smem_raw + input_byte_offset);
    } else {
        smem_input = reinterpret_cast<input_t *>(smem_raw);
    }

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    if constexpr (FilterInSmem) {
        // Load filter into smem
        if constexpr (FilterFullLayout) {
            // Full layout: one copy of each unique tap, laid out as
            //   smem_filter_base[p * M + phase] = filter[phase + p * M]
            // No per-channel duplication, no per-K duplication. At access
            // time the thread computes phase = (c + rotations[k]) % M and
            // reads smem_filter_base[p * M + phase].
            const int32_t total = P * M;
            for (int32_t i = tid; i < total; i += nthreads) {
                smem_filter_base[i] = (i < filter_full_len)
                    ? filter_acc(static_cast<index_t>(i))
                    : static_cast<filter_t>(0);
            }
        } else if constexpr (MaximallyDecimated) {
            for (int32_t i = tid; i < P * CTILE; i += nthreads) {
                const int32_t p  = i / CTILE;
                const int32_t local_channel = i % CTILE;
                const int32_t global_channel = tile_base + local_channel;
                const int32_t h_ind = global_channel + p * M;
                smem_filter_base[i] = (global_channel < M && h_ind < filter_full_len)
                    ? filter_acc(static_cast<index_t>(h_ind))
                    : static_cast<filter_t>(0);
            }
        } else {
            // The dispatch must not select FilterInSmem when K exceeds the
            // rotations[] array size. We rely on the transform dispatch to
            // ensure this invariant due to the cost of run-time kernel checks.
            int32_t rotations[detail::cpoly::SmemTiledMaxRotations];
            for (int32_t k = 0; k < K; k++) {
                rotations[k] = static_cast<int32_t>((static_cast<int64_t>(k) * decimation_factor) % M);
            }
            for (int32_t i = tid; i < CTILE * filter_stride; i += nthreads) {
                const int32_t local_channel = i / filter_stride;
                const int32_t kp = i % filter_stride;
                const int32_t k  = kp / P;
                const int32_t p  = kp % P;
                const int32_t global_channel = tile_base + local_channel;
                if (global_channel < M) {
                    // Phase uses the original logical channel (not remapped)
                    const int32_t phase = (global_channel + rotations[k]) % M;
                    const int32_t h_ind = phase + p * M;
                    smem_filter_base[i] = (h_ind < filter_full_len)
                        ? filter_acc(static_cast<index_t>(h_ind))
                        : static_cast<filter_t>(0);
                } else {
                    smem_filter_base[i] = static_cast<filter_t>(0);
                }
            }
        }
    }

    // r_remapped changes the input sample mapping to match the Harris convention. We conceptually populate commutator branches
    // starting at M-1 and continuing counter-clockwise. That matches the Harris convention for the maximally decimated
    // case where L == 0. For the oversampled case, Harris populates branches from M-1 to 0 for each M inputs, shifting
    // older samples through the 2D filter bank. We stick with the M-1 to 0 convention, but then have to remap the
    // branch indices to match the Harris convention.
    // Phase and output channel use the original logical index c.
    const int32_t r_remapped = (c + L) % M;
    const int32_t s = active ? (M - 1 - r_remapped) : 0;
    const IdxT start_elem = static_cast<IdxT>(blockIdx.x) * elems_per_channel_per_cta;
    const IdxT last_elem = cuda::std::min(
        output_len_per_channel - 1,
        start_elem + elems_per_channel_per_cta - 1);

    // Helper: load one input sample into smem at (buf_row, col)
    auto load_smem_elem = [&](int32_t buf_row, int32_t col, IdxT global_row) {
        const int32_t gc = tile_base + col;
        const int32_t gc_remapped = MaximallyDecimated ? gc : ((gc + L) % M);
        const int32_t branch_s = (gc < M) ? (M - 1 - gc_remapped) : 0;
        const IdxT raw_idx = static_cast<IdxT>(branch_s) + global_row * M;
        if (gc < M && global_row >= 0 && raw_idx >= 0 && raw_idx < input_len) {
            smem_input[buf_row * CTILE + col] = input_b(raw_idx);
        } else {
            smem_input[buf_row * CTILE + col] = static_cast<input_t>(0);
        }
    };

    // Converts a local output element to the newest global input-block index it
    // needs, folding in out_elem_offset so buffer bookkeeping and loads all key
    // off the global block index. All callers pass local output elements.
    auto max_bidx = [&](IdxT t_local) -> IdxT {
        const IdxT t = t_local + out_elem_offset;
        if constexpr (MaximallyDecimated) {
            return t;
        } else {
            return ((t + 1) * decimation_factor - 1) / M;
        }
    };

    // Initial fill of circular buffer
    const IdxT first_iter_end = cuda::std::min(start_elem + static_cast<IdxT>(NOUT) - 1, last_elem);
    IdxT loaded_up_to = max_bidx(first_iter_end);
    const IdxT buf_base = loaded_up_to - (height - 1);

    {
        const int32_t first_row = tid / CTILE;
        const int32_t row_stride = nthreads / CTILE;
        int32_t buf_row = static_cast<int32_t>((buf_base + first_row) % height);
        if (buf_row < 0) buf_row += height;
        const int32_t buf_row_stride = row_stride % height;

        for (int32_t i = tid; i < height * CTILE; i += nthreads) {
            load_smem_elem(buf_row, i % CTILE, buf_base + (i / CTILE));
            buf_row += buf_row_stride;
            if (buf_row >= height) buf_row -= height;
        }
    }

    __syncthreads();

    if constexpr (MaximallyDecimated) {
        // The dispatcher launches this path with NOUT=16. Critical 32x4 plans use the
        // oversampled path below with K=1.
        constexpr int YTHREADS = detail::cpoly::SmemTiledMaxDecYThreads;
        static_assert(NOUT % YTHREADS == 0);
        constexpr int OUTPUTS_PER_THREAD = NOUT / YTHREADS;
        const IdxT last_start = start_elem + ((last_elem - start_elem) / NOUT) * NOUT;
        for (IdxT next_start = start_elem;
             next_start <= last_start; next_start += NOUT) {
            accum_t accum[OUTPUTS_PER_THREAD]{};
            int32_t sample_row[OUTPUTS_PER_THREAD];
            bool valid[OUTPUTS_PER_THREAD];
            #pragma unroll
            for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                const IdxT t = next_start + ty + q * YTHREADS;
                valid[q] = t <= last_elem;
                const IdxT tg = t + out_elem_offset;
                sample_row[q] = static_cast<int32_t>(tg % height);
            }

            if (active) {
                int32_t filter_idx = (FilterInSmem && !FilterFullLayout) ? cx : c;
                for (int32_t p = 0; p < P; p++) {
                    filter_t hv;
                    if constexpr (FilterInSmem) {
                        hv = smem_filter_base[filter_idx];
                    } else {
                        hv = (filter_idx < filter_full_len)
                            ? filter_acc(static_cast<index_t>(filter_idx))
                            : static_cast<filter_t>(0);
                    }
                    const auto hav = detail::channelize_cast_operand<accum_t>(hv);
                    #pragma unroll
                    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                        const input_t iv = smem_input[sample_row[q] * CTILE + cx];
                        detail::channelize_cmac(
                            accum[q], hav, detail::channelize_cast_operand<accum_t>(iv));
                        if (--sample_row[q] < 0) {
                            sample_row[q] += height;
                        }
                    }
                    filter_idx += (FilterInSmem && !FilterFullLayout) ? CTILE : M;
                }
            }

            #pragma unroll
            for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                const IdxT t = next_start + ty + q * YTHREADS;
                if (active && valid[q]) {
                    output_b(t, c) = static_cast<output_t>(accum[q]);
                }
            }

            if (next_start < last_start) {
                // Ensure all threads have finished reading smem_input before overwriting it.
                __syncthreads();

                // Only load rows needed by this block. Speculative rows
                // beyond its final output can overflow 32-bit raw indices.
                const int32_t new_rows = static_cast<int32_t>(
                    cuda::std::min(static_cast<IdxT>(NOUT),
                                   last_elem - next_start - NOUT + 1));
                #pragma unroll
                for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                    const int32_t row = ty + q * YTHREADS;
                    if (row < new_rows) {
                        const IdxT global_row = next_start + out_elem_offset + NOUT + row;
                        const int32_t smem_row = static_cast<int32_t>(global_row % height);
                        load_smem_elem(smem_row, cx, global_row);
                    }
                }

                // Ensure new rows are visible before the next compute.
                __syncthreads();
            }
        }
    } else {
        // Oversampled path (D < M), also used by critical 32x4 plans.
        const IdxT last_start = start_elem + ((last_elem - start_elem) / NOUT) * NOUT;
        for (IdxT next_start = start_elem; next_start <= last_start; next_start += NOUT) {
            const IdxT t = next_start + ty;
            if (t <= last_elem && active) {
                accum_t accum{};

                const IdxT tg = t + out_elem_offset; // global output element index
                const IdxT last_arrived = tg * decimation_factor + decimation_factor - 1;
                if (last_arrived >= s) {
                    const IdxT A = last_arrived - s;
                    const IdxT bidx = A / M;
                    const IdxT causal_count = bidx + 1;

                    const int32_t k = static_cast<int32_t>(tg % K);
                    // Phase uses the original logical channel (not remapped)
                    const int32_t phase = static_cast<int32_t>(
                        (c + tg * decimation_factor) % M);

                    int32_t available_taps = P;
                    if (((P - 1) * M + phase) >= filter_full_len) {
                        available_taps--;
                    }

                    IdxT newest_raw = last_arrived - (A % M);
                    int32_t h_skip = 0;
                    if (newest_raw >= input_len) {
                        h_skip = 1;
                    }

                    const int32_t niter = static_cast<int32_t>(
                        cuda::std::min(static_cast<IdxT>(available_taps - h_skip),
                                       causal_count - h_skip));

                    int32_t buf_row = static_cast<int32_t>((bidx - h_skip) % height);

                    const int32_t prologue = cuda::std::min(buf_row + 1, niter);
                    const int32_t epilogue = niter - prologue;
                    // Single running counter instead of separate `p` and
                    // `h_ind`. Per layout (init, stride):
                    //   Full:    smem[(p+h_skip)*M + phase]         -> h_skip*M+phase, +M
                    //   Rotated: smem[cx*K*P + k*P + (p+h_skip)]    -> cx*K*P+k*P+h_skip, +1
                    //   Global:  filter[phase + (p+h_skip)*M]       -> phase+h_skip*M, +M
                    int32_t filter_idx;
                    if constexpr (FilterInSmem && !FilterFullLayout) {
                        filter_idx = cx * filter_stride + k * P + h_skip;
                    } else {
                        filter_idx = phase + h_skip * M;
                    }
                    #pragma unroll 8
                    for (int32_t i = 0; i < prologue; i++) {
                        filter_t hv;
                        if constexpr (FilterInSmem) {
                            hv = smem_filter_base[filter_idx];
                        } else {
                            hv = filter_acc(static_cast<index_t>(filter_idx));
                        }
                        const input_t iv = smem_input[buf_row * CTILE + cx];
                        detail::channelize_cmac(accum,
                                detail::channelize_cast_operand<accum_t>(hv),
                                detail::channelize_cast_operand<accum_t>(iv));
                        buf_row--;
                        if constexpr (FilterInSmem && !FilterFullLayout) {
                            filter_idx += 1;
                        } else {
                            filter_idx += M;
                        }
                    }
                    buf_row = height - 1;
                    #pragma unroll 8
                    for (int32_t i = 0; i < epilogue; i++) {
                        filter_t hv;
                        if constexpr (FilterInSmem) {
                            hv = smem_filter_base[filter_idx];
                        } else {
                            hv = filter_acc(static_cast<index_t>(filter_idx));
                        }
                        const input_t iv = smem_input[buf_row * CTILE + cx];
                        detail::channelize_cmac(accum,
                                detail::channelize_cast_operand<accum_t>(hv),
                                detail::channelize_cast_operand<accum_t>(iv));
                        buf_row--;
                        if constexpr (FilterInSmem && !FilterFullLayout) {
                            filter_idx += 1;
                        } else {
                            filter_idx += M;
                        }
                    }
                }

                output_b(t, c) = static_cast<output_t>(accum);
            }

            if (next_start < last_start) {
                // Ensure all threads have finished reading smem_input before overwriting
                __syncthreads();

                // Load new rows for next iteration
                const IdxT next_iter_end = cuda::std::min(next_start + static_cast<IdxT>(2 * NOUT) - 1, last_elem);
                const IdxT needed_up_to = max_bidx(next_iter_end);
                const int32_t new_rows = static_cast<int32_t>(needed_up_to - loaded_up_to);

                // Each ty-lane loads one row (its cx column)
                {
                    int32_t lr = static_cast<int32_t>((loaded_up_to + 1) % height);
                    if (ty < new_rows) {
                        int32_t my_lr = lr + ty;
                        if (my_lr >= height) my_lr -= height;
                        load_smem_elem(my_lr, cx, loaded_up_to + 1 + ty);
                    }
                }

                loaded_up_to = needed_up_to;

                // Ensure new rows are visible before next iteration's compute
                __syncthreads();
            }
        }
    }
}

// This kernel works in cases where the full filter (with potentially some zero padding) and
// the inputs required to compute elems_per_channel_per_cta outputs all fit into shared memory.
// See ChannelizePoly1D for the out_elem_offset output-element window semantics.
// Here only the two input-staging loads consult the global output-element index
// (out_sample_ind / input_ind); the write row (out_elem_idx) stays local.
template <bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename AccumType>
__global__ void ChannelizePoly1D_Smem(OutType output, InType input, FilterType filter, index_t elems_per_channel_per_cta, index_t out_elem_offset)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;
    static_assert(! is_complex_v<AccumType>,
        "channelize_poly: accumulator type must be real; it will be treated as complex when necessary");
    // If the output is complex, then accumulator is complex. Otherwise, the accumulator is real.
    using accum_t = cuda::std::conditional_t<is_complex_v<output_t>, typename detail::scalar_to_complex<AccumType>::ctype, AccumType>;

    extern __shared__ __align__(16) uint8_t smem_dyn_align16[];

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int ChannelRank = OutRank-1;
    constexpr int OutElemRank = OutRank-2;

    const index_t input_len = input.Size(InRank-1);
    const index_t output_len_per_channel = output.Size(OutElemRank);
    // If the filter fits into shared memory, then a 32-bit index is sufficient. One
    // edge case exception would be num_channels > 2^32-1, but with a small filter
    // implicitly padded with zeros. We assume that the kernel selection logic
    // considers the size of the zero-padded filter since that is what we actually
    // store in shared memory.
    const int32_t num_channels = static_cast<int32_t>(output.Size(ChannelRank));
    const int32_t filter_full_len = static_cast<int32_t>(filter.Size(0));
    const int32_t filter_phase_len = static_cast<int32_t>((filter_full_len + num_channels - 1) / num_channels);

    filter_t *smem_h = reinterpret_cast<filter_t *>(smem_dyn_align16);
    size_t smem_input_offset = sizeof(filter_t) * filter_phase_len * num_channels;
    if (smem_input_offset % sizeof(input_t)) {
        smem_input_offset += sizeof(input_t) - smem_input_offset % sizeof(input_t);
    }
    input_t *smem_input = reinterpret_cast<input_t *>(smem_dyn_align16 + smem_input_offset);

    const int32_t tid = static_cast<int32_t>(threadIdx.y * blockDim.x + threadIdx.x);
    const int32_t nthreads = static_cast<int32_t>(blockDim.x * blockDim.y);
    const int32_t chan = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);
    const int32_t by = static_cast<int32_t>(blockDim.y);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    for (int32_t t = tid; t < filter_full_len; t += nthreads) {
        smem_h[t] = filter_acc(t);
    }

    for (int32_t t = filter_full_len+tid; t < filter_phase_len * num_channels; t += nthreads) {
        smem_h[t] = static_cast<filter_t>(0);
    }

    // The input stored in shared memory is logically [smem_input_height, num_channels] where
    // smem_input_height is the number of samples at the output sample rate stored in smem.
    const int32_t smem_input_height = filter_phase_len + by - 1;

    const index_t start_elem = blockIdx.x * elems_per_channel_per_cta;
    const index_t last_elem_this_block = static_cast<index_t>(blockIdx.x) * elems_per_channel_per_cta + (elems_per_channel_per_cta - 1);
    const index_t last_elem = cuda::std::min(output_len_per_channel-1, last_elem_this_block);

    for (int32_t t = ty; t < filter_phase_len-1; t += by) {
        const index_t out_sample_ind = start_elem + out_elem_offset - (filter_phase_len-1) + t;
        const int32_t smem_ind = t * num_channels + chan;
        const index_t input_ind = out_sample_ind * num_channels + chan;
        if (input_ind >= 0 && input_ind < input_len) {
            smem_input[smem_ind] = input_b(input_ind);
        } else {
            smem_input[smem_ind] = static_cast<input_t>(0);
        }
    }

    index_t next_start_elem = start_elem;
    const index_t num_elem_iters = (last_elem - start_elem + 1 + by - 1) / by;

    int32_t cached_input_ind_tail = filter_phase_len - 1 + ty;
    const filter_t *h_start = smem_h + num_channels * filter_phase_len - (num_channels - chan);
    for (index_t iter = 0; iter < num_elem_iters; iter++) {

        __syncthreads();

        // Load next elems_per_channel_per_cta elements for each channel
        const index_t next_last_elem = cuda::std::min(next_start_elem + static_cast<index_t>(by) - 1, last_elem);
        const int32_t out_samples_this_iter = static_cast<int32_t>(next_last_elem - next_start_elem + 1);
        if (ty < out_samples_this_iter) {
            const index_t input_ind = (next_start_elem + out_elem_offset + ty) * num_channels + chan;
            const int32_t smem_ind = cached_input_ind_tail * num_channels + chan;
            if (input_ind < input_len) {
                smem_input[smem_ind] = input_b(input_ind);
            } else {
                smem_input[smem_ind] = static_cast<input_t>(0);
            }
        }

        cached_input_ind_tail += by;
        // The below effectively mods cached_input_ind_tail by smem_input_height. Since
        // smem_input_height is >= by, adding by means that we will need to subtract
        // smem_input_height at most once for cached_input_ind_tail to be in the range
        // [0, smem_input_height-1]. The conditional is cheaper than the mod, unless
        // smem_input_height is known at compile time.
        if (cached_input_ind_tail >= smem_input_height) {
            cached_input_ind_tail -= smem_input_height;
        }

        __syncthreads();

        const index_t out_elem_idx = next_start_elem + ty;
        if (out_elem_idx <= last_elem) {
            const filter_t *h = h_start;
            accum_t accum { 0 };
            const int32_t first_end = cuda::std::min(cached_input_ind_tail + filter_phase_len - 1, smem_input_height - 1);
            // The footprint of samples involved in the convolution may wrap from the end
            // to the beginning of smem_input. The prologue below handles the samples from
            // the current tail to the end of smem_input and the epilogue starts back at the
            // beginning of smem_input.
            const int32_t prologue_count = (first_end - cached_input_ind_tail + 1);
            const int32_t epilogue_count = (prologue_count < filter_phase_len) ? filter_phase_len - prologue_count : 0;
            const input_t *sample = smem_input + cached_input_ind_tail * num_channels + (num_channels - 1 - chan);
            // Apply the filter h in reverse order below to flip the filter for convolution
            for (int32_t k = 0; k < prologue_count; k++) {
                detail::channelize_cmac(accum,
                        detail::channelize_cast_operand<accum_t>(*h),
                        detail::channelize_cast_operand<accum_t>(*sample));
                sample += num_channels;
                h -= num_channels;
            }
            sample = smem_input + (num_channels - 1 - chan);
            for (int32_t k = 0; k < epilogue_count; k++) {
                detail::channelize_cmac(accum,
                        detail::channelize_cast_operand<accum_t>(*h),
                        detail::channelize_cast_operand<accum_t>(*sample));
                sample += num_channels;
                h -= num_channels;
            }

            output_b(out_elem_idx, chan) = static_cast<output_t>(accum);
        }

        next_start_elem += out_samples_this_iter;
    }
}

template <int NUM_CHAN, typename OutType, typename InType, typename FilterType, typename AccumType>
struct ChannelizePolyFusedSmallTraits {
    using input_t = typename InType::value_type;
    using output_t = typename OutType::value_type;
    using filter_t = typename FilterType::value_type;
    using complex_accum_t = typename detail::scalar_to_complex<AccumType>::ctype;
    using accum_t = cuda::std::conditional_t<
        is_complex_v<input_t> || is_complex_v<filter_t>, complex_accum_t, AccumType>;
    static constexpr bool ComplexInput = is_complex_v<input_t>;
    static constexpr bool UseM2PairLeaf =
        (cuda::std::is_same_v<AccumType, float> ||
         cuda::std::is_same_v<AccumType, double>) &&
        !is_complex_v<filter_t> && cuda::std::is_same_v<filter_t, AccumType> &&
        (cuda::std::is_same_v<input_t, AccumType> ||
         cuda::std::is_same_v<input_t, complex_accum_t>);

    static_assert(NUM_CHAN >= 2 && NUM_CHAN <= 6);
    static_assert(is_complex_v<output_t>);
    static_assert(!is_complex_v<AccumType>);
};

template <typename Complex, typename Real, typename Imag>
__device__ __forceinline__ Complex ChannelizePolyFusedSmallComplex(Real real, Imag imag)
{
    using scalar_t = typename inner_op_type_t<Complex>::type;
    return Complex{static_cast<scalar_t>(real), static_cast<scalar_t>(imag)};
}

template <int THREADS, typename FilterAccessor, typename Scalar>
__device__ __forceinline__ void ChannelizePolyFusedM2LoadFilter(
    FilterAccessor &filter, detail::ChannelizePolyM2Pair<Scalar> *smem_filter,
    int32_t taps_per_channel, index_t filter_len, int32_t tid)
{
    for (int32_t p = tid; p < taps_per_channel; p += THREADS) {
        const index_t h = static_cast<index_t>(p) * 2;
        smem_filter[p] = {
            filter(h), (h + 1 < filter_len) ? filter(h + 1) : Scalar{0}};
    }
}

template <int THREADS, int OUTPUTS_PER_THREAD, typename OutputAccessor, typename Scalar>
__device__ __forceinline__ void ChannelizePolyFusedM2Store(
    OutputAccessor &output, index_t output_len, index_t block_start,
    int32_t tid, const Scalar (&a0r)[OUTPUTS_PER_THREAD], const Scalar (&a0i)[OUTPUTS_PER_THREAD],
    const Scalar (&a1r)[OUTPUTS_PER_THREAD], const Scalar (&a1i)[OUTPUTS_PER_THREAD])
{
    #pragma unroll
    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
        const index_t t = block_start + tid + q * THREADS;
        if (t < output_len) {
            output(t, static_cast<index_t>(0)) =
                cuda::std::complex<Scalar>{a0r[q] + a1r[q],
                                           a0i[q] + a1i[q]};
            output(t, static_cast<index_t>(1)) =
                cuda::std::complex<Scalar>{a0r[q] - a1r[q],
                                           a0i[q] - a1i[q]};
        }
    }
}

template <typename Scalar>
__device__ __forceinline__ cuda::std::complex<Scalar>
ChannelizePolyTwiddle(int32_t numerator, int32_t denominator)
{
    const Scalar angle = static_cast<Scalar>(2) *
        static_cast<Scalar>(M_PI) * static_cast<Scalar>(numerator) /
        static_cast<Scalar>(denominator);
    Scalar sinx, cosx;
    if constexpr (cuda::std::is_same_v<Scalar, double>) {
        sincos(angle, &sinx, &cosx);
    } else {
        sincosf(angle, &sinx, &cosx);
    }
    return {cosx, sinx};
}

// A quarter wave covers every radix-2 root in the current dispatch (up to 64).
// Store native-precision constants once instead of evaluating sincos per CTA.
template <typename Scalar>
static __device__ __constant__ const Scalar ChannelizePolyRadix2Cos[] = {
    static_cast<Scalar>(1.0),                 // cos(0)
    static_cast<Scalar>(0.99518472667219693), // cos(pi/32)
    static_cast<Scalar>(0.98078528040323043), // cos(2*pi/32)
    static_cast<Scalar>(0.95694033573220882), // cos(3*pi/32)
    static_cast<Scalar>(0.92387953251128674), // cos(4*pi/32)
    static_cast<Scalar>(0.88192126434835505), // cos(5*pi/32)
    static_cast<Scalar>(0.83146961230254524), // cos(6*pi/32)
    static_cast<Scalar>(0.77301045336273699), // cos(7*pi/32)
    static_cast<Scalar>(0.70710678118654757), // cos(8*pi/32)
    static_cast<Scalar>(0.63439328416364549), // cos(9*pi/32)
    static_cast<Scalar>(0.55557023301960218), // cos(10*pi/32)
    static_cast<Scalar>(0.47139673682599764), // cos(11*pi/32)
    static_cast<Scalar>(0.38268343236508978), // cos(12*pi/32)
    static_cast<Scalar>(0.29028467725446239), // cos(13*pi/32)
    static_cast<Scalar>(0.19509032201612828), // cos(14*pi/32)
    static_cast<Scalar>(0.098017140329560604), // cos(15*pi/32)
    static_cast<Scalar>(0.0),                 // cos(pi/2)
};

template <typename Scalar, int RADIX_2>
__device__ __forceinline__ cuda::std::complex<Scalar>
ChannelizePolyRadix2Twiddle(int32_t j)
{
    if constexpr (RADIX_2 <= 64) {
        const int32_t phase = j * (64 / RADIX_2);
        const int32_t cos_index = phase <= 16 ? phase : 32 - phase;
        const int32_t sin_index = phase <= 16 ? 16 - phase : phase - 16;
        const Scalar cosx = ChannelizePolyRadix2Cos<Scalar>[cos_index];
        const Scalar sinx = ChannelizePolyRadix2Cos<Scalar>[sin_index];
        return {phase <= 16 ? cosx : -cosx, sinx};
    } else {
        // Keep larger compile-time radices usable without expanding the table.
        return ChannelizePolyTwiddle<Scalar>(j, RADIX_2);
    }
}

template <typename Complex>
__device__ __forceinline__ void ChannelizePolyRadix3(
    const Complex &x0, const Complex &x1, const Complex &x2, Complex &y0, Complex &y1, Complex &y2)
{
    using scalar_t = typename inner_op_type_t<Complex>::type;
    const scalar_t half = static_cast<scalar_t>(0.5); // cos(pi/3)
    const scalar_t sin60 = static_cast<scalar_t>(0.8660254037844386); // sin(pi/3)
    const Complex sum = x1 + x2;
    const Complex diff = x1 - x2;
    const Complex base = x0 - half * sum;
    const Complex jdiff = ChannelizePolyFusedSmallComplex<Complex>(
        -sin60 * diff.imag(), sin60 * diff.real());
    y0 = x0 + sum;
    y1 = base + jdiff;
    y2 = base - jdiff;
}

template <typename Complex>
__device__ __forceinline__ void ChannelizePolyRadix5(
    const Complex &x0, const Complex &x1, const Complex &x2,
    const Complex &x3, const Complex &x4, Complex &y0, Complex &y1,
    Complex &y2, Complex &y3, Complex &y4)
{
    using scalar_t = typename inner_op_type_t<Complex>::type;
    const scalar_t c1 = static_cast<scalar_t>(0.30901699437494745); // cos(2*pi/5)
    const scalar_t c2 = static_cast<scalar_t>(-0.80901699437494745); // cos(4*pi/5)
    const scalar_t s1 = static_cast<scalar_t>(0.95105651629515357); // sin(2*pi/5)
    const scalar_t s2 = static_cast<scalar_t>(0.58778525229247314); // sin(4*pi/5)
    const Complex sum14 = x1 + x4;
    const Complex diff14 = x1 - x4;
    const Complex sum23 = x2 + x3;
    const Complex diff23 = x2 - x3;
    const Complex base1 = x0 + c1 * sum14 + c2 * sum23;
    const Complex base2 = x0 + c2 * sum14 + c1 * sum23;
    const Complex odd1 = s1 * diff14 + s2 * diff23;
    const Complex odd2 = s2 * diff14 - s1 * diff23;
    const Complex jodd1 = ChannelizePolyFusedSmallComplex<Complex>(-odd1.imag(), odd1.real());
    const Complex jodd2 = ChannelizePolyFusedSmallComplex<Complex>(-odd2.imag(), odd2.real());
    y0 = x0 + sum14 + sum23;
    y1 = base1 + jodd1;
    y2 = base2 + jodd2;
    y3 = base2 - jodd2;
    y4 = base1 - jodd1;
}

// Positive-sign, unnormalized DFT used by the channelizer. These fixed
// butterflies replace FusedChan's per-CTA sincos table and quadratic DFT for
// the small channel counts that it historically handled.
template <int NUM_CHAN, typename Complex>
__device__ __forceinline__ void ChannelizePolySmallDFT(
    const Complex (&x)[NUM_CHAN], Complex (&y)[NUM_CHAN])
{
    using scalar_t = typename inner_op_type_t<Complex>::type;
    static_assert(NUM_CHAN >= 2 && NUM_CHAN <= 6);
    if constexpr (NUM_CHAN == 2) {
        y[0] = x[0] + x[1];
        y[1] = x[0] - x[1];
    } else if constexpr (NUM_CHAN == 3) {
        ChannelizePolyRadix3(x[0], x[1], x[2], y[0], y[1], y[2]);
    } else if constexpr (NUM_CHAN == 4) {
        Complex sum[2], diff[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            sum[i] = x[i] + x[i + 2];
            diff[i] = x[i] - x[i + 2];
        }
        const Complex jdiff1 = ChannelizePolyFusedSmallComplex<Complex>(
            -diff[1].imag(), diff[1].real());
        y[0] = sum[0] + sum[1];
        y[1] = diff[0] + jdiff1;
        y[2] = sum[0] - sum[1];
        y[3] = diff[0] - jdiff1;
    } else if constexpr (NUM_CHAN == 5) {
        ChannelizePolyRadix5(x[0], x[1], x[2], x[3], x[4], y[0], y[1], y[2], y[3], y[4]);
    } else {
        const scalar_t half = static_cast<scalar_t>(0.5); // cos(pi/3)
        const scalar_t sin60 = static_cast<scalar_t>(0.8660254037844386); // sin(pi/3)
        Complex even[3], odd[3];
        ChannelizePolyRadix3(x[0], x[2], x[4], even[0], even[1], even[2]);
        ChannelizePolyRadix3(x[1], x[3], x[5], odd[0], odd[1], odd[2]);
        Complex twiddled[3];
        twiddled[0] = odd[0];
        twiddled[1] = ChannelizePolyFusedSmallComplex<Complex>(half, sin60) * odd[1];
        twiddled[2] = ChannelizePolyFusedSmallComplex<Complex>(-half, sin60) * odd[2];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            y[i] = even[i] + twiddled[i];
            y[i + 3] = even[i] - twiddled[i];
        }
    }
}

template <int THREADS, int NUM_CHAN, int OUTPUTS_PER_THREAD,
          typename OutputAccessor, typename AccumType>
__device__ __forceinline__ void ChannelizePolyFusedSmallTransformStore(
    OutputAccessor &output, index_t output_len, index_t block_start, int32_t tid,
    const AccumType (&accum)[OUTPUTS_PER_THREAD][NUM_CHAN])
{
    using scalar_t = typename inner_op_type_t<AccumType>::type;
    using complex_t = typename detail::scalar_to_complex<scalar_t>::ctype;
    static_assert(cuda::std::is_same_v<AccumType, scalar_t> ||
                  cuda::std::is_same_v<AccumType, complex_t>);
    #pragma unroll
    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
        const index_t t = block_start + tid + q * THREADS;
        if (t < output_len) {
            complex_t transformed[NUM_CHAN];
            complex_t dft_input[NUM_CHAN];
            #pragma unroll
            for (int c = 0; c < NUM_CHAN; c++) {
                dft_input[c] = static_cast<complex_t>(accum[q][c]);
            }
            ChannelizePolySmallDFT(dft_input, transformed);
            #pragma unroll
            for (int c = 0; c < NUM_CHAN; c++) {
                output(t, static_cast<index_t>(c)) = transformed[c];
            }
        }
    }
}

// Critically sampled row-owned leaf for the small-M portion of the unified
// fused radix backend. A CTA stages one overlapping FIR tile; each thread then
// computes every branch and the fixed K-by-power-of-two DFT for its rows.
template <int THREADS, int NUM_CHAN, int OUTPUTS_PER_THREAD,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType>
__device__ __forceinline__ void ChannelizePolyFusedSmallCachedBody(
    OutType &output, const InType &input, const FilterType &filter,
    index_t out_elem_offset, uint8_t *smem_raw)
{
    constexpr int BLOCK_ROWS = THREADS * OUTPUTS_PER_THREAD;
    using traits = ChannelizePolyFusedSmallTraits<NUM_CHAN, OutType, InType, FilterType, AccumType>;
    using input_t = typename traits::input_t;
    using filter_t = typename traits::filter_t;
    using accum_t = typename traits::accum_t;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank - 2;
    const index_t input_len = input.Size(InRank - 1);
    const index_t output_len = output.Size(OutElemRank);
    const index_t filter_len = filter.Size(0);
    const int32_t P = static_cast<int32_t>((filter_len + NUM_CHAN - 1) / NUM_CHAN);

    const detail::cpoly::FusedRadixSmemLayout<
        NUM_CHAN, true, input_t, filter_t, AccumType> layout(P, NUM_CHAN);
    filter_t *smem_filter = reinterpret_cast<filter_t *>(smem_raw);
    input_t *smem_input = reinterpret_cast<input_t *>(smem_raw + layout.input_offset);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t filter_elements = static_cast<int32_t>(layout.filter_elements);
    for (int32_t i = tid; i < filter_elements; i += THREADS) {
        smem_filter[i] = (i < filter_len)
            ? filter_acc(static_cast<index_t>(i)) : filter_t{};
    }

    const index_t block_start = static_cast<index_t>(blockIdx.x) * BLOCK_ROWS;
    const index_t input_base = (block_start + out_elem_offset - (P - 1)) * NUM_CHAN;
    const int32_t input_elements = static_cast<int32_t>(layout.input_elements);
    for (int32_t i = tid; i < input_elements; i += THREADS) {
        // Linear staging preserves fully coalesced global loads. This is faster than transposing
        // the tile while it is staged, despite the modest shared-memory conflicts during FIR reuse.
        smem_input[i] = detail::cpoly::LoadRelativeInput(input_b, input_base, input_len, i);
    }
    __syncthreads();

    accum_t accum[OUTPUTS_PER_THREAD][NUM_CHAN]{};
    const index_t t = block_start + tid;
    if (t < output_len) {
        const int32_t output_row = P - 1 + tid;
        for (int32_t p = 0; p < P; p++) {
            #pragma unroll
            for (int c = 0; c < NUM_CHAN; c++) {
                // Reuse each filter coefficient across the output accumulators.
                const filter_t hv = smem_filter[p * NUM_CHAN + c];
                #pragma unroll
                for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
                    if (q == 0 || t + q * THREADS < output_len) {
                        const input_t xv = smem_input[
                            (output_row + q * THREADS - p) * NUM_CHAN + (NUM_CHAN - 1 - c)];
                        detail::channelize_cmac(
                            accum[q][c], detail::channelize_cast_operand<accum_t>(hv),
                            detail::channelize_cast_operand<accum_t>(xv));
                    }
                }
            }
        }
    }

    ChannelizePolyFusedSmallTransformStore<
        THREADS, NUM_CHAN, OUTPUTS_PER_THREAD>(
            output_b, output_len, block_start, tid, accum);
}

// Direct short-filter leaf of the same small-M backend. This deliberately shares the enclosing
// global-kernel instantiation with the cached leaf: P is a uniform runtime choice, so supporting it
// does not add another exported CUDA kernel or another host dispatch specialization.
template <int THREADS, int NUM_CHAN, int OUTPUTS_PER_THREAD,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType>
__device__ __forceinline__ void ChannelizePolyFusedSmallDirectBody(
    OutType &output, const InType &input, const FilterType &filter, index_t out_elem_offset)
{
    constexpr int BLOCK_ROWS = THREADS * OUTPUTS_PER_THREAD;
    using traits = ChannelizePolyFusedSmallTraits<NUM_CHAN, OutType, InType, FilterType, AccumType>;
    using input_t = typename traits::input_t;
    using filter_t = typename traits::filter_t;
    using accum_t = typename traits::accum_t;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank - 2;
    const index_t input_len = input.Size(InRank - 1);
    const index_t output_len = output.Size(OutElemRank);
    const index_t filter_len = filter.Size(0);
    const int32_t P = static_cast<int32_t>((filter_len + NUM_CHAN - 1) / NUM_CHAN);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const index_t block_start = static_cast<index_t>(blockIdx.x) * BLOCK_ROWS;
    if constexpr (NUM_CHAN == 2 && cuda::std::is_same_v<AccumType, double>) {
        // This leaf has at most four taps per channel; fixed loops avoid
        // runtime FIR-loop overhead for double accumulation.
        #pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
            accum_t accum[1][NUM_CHAN]{};
            const index_t t = block_start + tid + q * THREADS;
            if (t < output_len) {
                const index_t g = t + out_elem_offset;
                const int32_t last = static_cast<int32_t>(
                    cuda::std::min(static_cast<index_t>(P), g + 1));
                constexpr int max_taps = detail::cpoly::FusedSmallDirectMaxTaps<
                    NUM_CHAN, input_t, AccumType>();
                #pragma unroll
                for (int32_t p = 0; p < max_taps; p++) {
                    if (p >= last) continue;
                    #pragma unroll
                    for (int c = 0; c < NUM_CHAN; c++) {
                        const index_t h = static_cast<index_t>(p) * NUM_CHAN + c;
                        const index_t x = (g - p) * NUM_CHAN + NUM_CHAN - 1 - c;
                        if (h >= filter_len || x >= input_len) continue;
                        detail::channelize_cmac(
                            accum[0][c], detail::channelize_cast_operand<accum_t>(filter_acc(h)),
                            detail::channelize_cast_operand<accum_t>(input_b(x)));
                    }
                }
            }
            ChannelizePolyFusedSmallTransformStore<THREADS, NUM_CHAN, 1>(
                output_b, output_len, block_start + q * THREADS, tid, accum);
        }
        return;
    }
    accum_t accum[OUTPUTS_PER_THREAD][NUM_CHAN]{};
    #pragma unroll
    for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
        const index_t t = block_start + tid + q * THREADS;
        if (t < output_len) {
            const index_t g = t + out_elem_offset;
            constexpr bool peel_bounds =
                detail::cpoly::FusedSmallDirectMaxTaps<
                    NUM_CHAN, input_t, AccumType>() > 4;
            const int32_t last = peel_bounds ? static_cast<int32_t>(
                cuda::std::min(static_cast<index_t>(P), g + 1)) : P;
            for (int32_t p = 0; p < last; p++) {
                const index_t input_row = g - p;
                if constexpr (!peel_bounds) {
                    if (input_row < 0) continue;
                }
                const index_t input_base = input_row * NUM_CHAN;
                // Interior taps have complete input and filter rows. Preserve
                // short-direct scheduling for the other types/channel counts.
                if (peel_bounds && p > 0 && p + 1 < P) {
                    #pragma unroll
                    for (int c = 0; c < NUM_CHAN; c++) {
                        const filter_t hv = filter_acc(static_cast<index_t>(p) * NUM_CHAN + c);
                        const input_t xv = input_b(input_base + NUM_CHAN - 1 - c);
                        detail::channelize_cmac(
                            accum[q][c], detail::channelize_cast_operand<accum_t>(hv),
                            detail::channelize_cast_operand<accum_t>(xv));
                    }
                    continue;
                }
                #pragma unroll
                for (int c = 0; c < NUM_CHAN; c++) {
                    const index_t h = static_cast<index_t>(p) * NUM_CHAN + c;
                    const index_t x = input_base + NUM_CHAN - 1 - c;
                    if (h < filter_len && x < input_len) {
                        const filter_t hv = filter_acc(h);
                        const input_t xv = input_b(x);
                        detail::channelize_cmac(
                            accum[q][c], detail::channelize_cast_operand<accum_t>(hv),
                            detail::channelize_cast_operand<accum_t>(xv));
                    }
                }
            }
        }
    }

    ChannelizePolyFusedSmallTransformStore<
        THREADS, NUM_CHAN, OUTPUTS_PER_THREAD>(
            output_b, output_len, block_start, tid, accum);
}

// Maximally-decimated two-channel leaf of the fused radix/power-of-two
// family. A CTA stages a contiguous input window once, then reuses it across
// all FIR outputs before applying the exact two-point inverse DFT.
template <int THREADS, int OUTPUTS_PER_THREAD, bool IsUnitStride,
          typename OutType, typename InType, typename FilterType, typename AccumType>
__device__ __forceinline__ void ChannelizePolyFusedM2D2Body(
    OutType &output, const InType &input, const FilterType &filter,
    index_t out_elem_offset, index_t elems_per_channel_per_cta, uint8_t *smem_raw)
{
    constexpr int BLOCK_ROWS = THREADS * OUTPUTS_PER_THREAD;
    constexpr int NUM_CHAN = 2;
    using traits = ChannelizePolyFusedSmallTraits<NUM_CHAN, OutType, InType, FilterType, AccumType>;
    using input_t = typename traits::input_t;
    using scalar_t = AccumType;
    using filter_pair_t = detail::ChannelizePolyM2Pair<scalar_t>;
    constexpr bool ComplexInput = traits::ComplexInput;
    using shared_input_t = detail::ChannelizePolyM2Pair<input_t>;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank - 2;
    const index_t input_len = input.Size(InRank - 1);
    const index_t output_len = output.Size(OutElemRank);
    const index_t filter_len = filter.Size(0);
    const int32_t P = static_cast<int32_t>((filter_len + 1) / NUM_CHAN);

    // Keep the maximum shared layout, but stage only the active output span.
    const int32_t block_rows = ComplexInput &&
        cuda::std::is_same_v<AccumType, float>
        ? static_cast<int32_t>(elems_per_channel_per_cta) : BLOCK_ROWS;

    if constexpr (cuda::std::is_same_v<scalar_t, double>) {
        if (detail::cpoly::FusedSmallUseDirect<
                NUM_CHAN, OutType, InType, FilterType, AccumType>(P, output_len)) {
            // Keep the usual small shared allocation and launch geometry even
            // when bypassing staging, so this needs no separate host policy.
            ChannelizePolyFusedSmallDirectBody<
                THREADS, NUM_CHAN, OUTPUTS_PER_THREAD, IsUnitStride,
                OutType, InType, FilterType, AccumType>(
                    output, input, filter, out_elem_offset);
            return;
        }
    }

    const detail::cpoly::FusedRadixSmemLayout<
        NUM_CHAN, true, input_t, scalar_t, AccumType> layout(P, NUM_CHAN);
    filter_pair_t *smem_filter = reinterpret_cast<filter_pair_t *>(smem_raw);
    shared_input_t *smem_input = reinterpret_cast<shared_input_t *>(smem_raw + layout.input_offset);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    ChannelizePolyFusedM2LoadFilter<THREADS>(filter_acc, smem_filter, P, filter_len, tid);

    const index_t block_start = static_cast<index_t>(blockIdx.x) * block_rows;
    const index_t input_base = (block_start + out_elem_offset - (P - 1)) * NUM_CHAN;
    const int32_t input_rows =
        static_cast<int32_t>(layout.input_elements / NUM_CHAN) - (BLOCK_ROWS - block_rows);
    for (int32_t row = tid; row < input_rows; row += THREADS) {
        const int32_t offset = row * NUM_CHAN;
        const input_t even = detail::cpoly::LoadRelativeInput(
            input_b, input_base, input_len, offset);
        const input_t odd = detail::cpoly::LoadRelativeInput(
            input_b, input_base, input_len, offset + 1);
        smem_input[row] = {even, odd};
    }
    __syncthreads();

    scalar_t a0r[OUTPUTS_PER_THREAD]{};
    scalar_t a0i[OUTPUTS_PER_THREAD]{};
    scalar_t a1r[OUTPUTS_PER_THREAD]{};
    scalar_t a1i[OUTPUTS_PER_THREAD]{};
    for (int32_t p = 0; p < P; p++) {
        const filter_pair_t hv = smem_filter[p];
        #pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
            if (q > 0 && q * THREADS >= block_rows) continue;
            const int32_t row = (P - 1) + tid + q * THREADS - p;
            const shared_input_t xv = smem_input[row];
            if constexpr (ComplexInput) {
                a0r[q] = hv.even * xv.odd.real() + a0r[q];
                a0i[q] = hv.even * xv.odd.imag() + a0i[q];
                a1r[q] = hv.odd * xv.even.real() + a1r[q];
                a1i[q] = hv.odd * xv.even.imag() + a1i[q];
            } else {
                a0r[q] = hv.even * xv.odd + a0r[q];
                a1r[q] = hv.odd * xv.even + a1r[q];
            }
        }
    }

    const index_t store_end = ComplexInput &&
        cuda::std::is_same_v<AccumType, float>
        ? cuda::std::min(output_len, block_start + block_rows) : output_len;
    ChannelizePolyFusedM2Store<THREADS, OUTPUTS_PER_THREAD>(
        output_b, store_end, block_start, tid, a0r, a0i, a1r, a1i);
}

// Two-channel, decimation-one leaf of the fused radix/power-of-two family.
// Each output alternates the two filter phases while advancing by one raw
// input sample. Staging a contiguous raw-input window avoids launching the
// mostly-idle generic channel-tiled kernel for only two active channels.
template <int THREADS, int OUTPUTS_PER_THREAD, bool IsUnitStride,
          typename OutType, typename InType, typename FilterType, typename AccumType>
__device__ __forceinline__ void ChannelizePolyFusedM2D1Body(
    OutType &output, const InType &input, const FilterType &filter, index_t out_elem_offset,
    uint8_t *smem_raw)
{
    constexpr int BLOCK_OUTPUTS = THREADS * OUTPUTS_PER_THREAD;
    constexpr int NUM_CHAN = 2;
    using traits = ChannelizePolyFusedSmallTraits<NUM_CHAN, OutType, InType, FilterType, AccumType>;
    using input_t = typename traits::input_t;
    using scalar_t = AccumType;
    using filter_pair_t = detail::ChannelizePolyM2Pair<scalar_t>;
    constexpr bool ComplexInput = traits::ComplexInput;
    using shared_input_t = input_t;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank - 2;
    const index_t input_len = input.Size(InRank - 1);
    const index_t output_len = output.Size(OutElemRank);
    const index_t filter_len = filter.Size(0);
    const int32_t P = static_cast<int32_t>((filter_len + 1) / NUM_CHAN);

    const detail::cpoly::FusedRadixSmemLayout<
        NUM_CHAN, false, input_t, scalar_t, AccumType> layout(P, 1);
    filter_pair_t *smem_filter = reinterpret_cast<filter_pair_t *>(smem_raw);
    shared_input_t *smem_input = reinterpret_cast<shared_input_t *>(smem_raw + layout.input_offset);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    ChannelizePolyFusedM2LoadFilter<THREADS>(filter_acc, smem_filter, P, filter_len, tid);

    const index_t block_start = static_cast<index_t>(blockIdx.x) * BLOCK_OUTPUTS;
    const index_t input_base = block_start + out_elem_offset - 2 * P;
    const int32_t staged_inputs = static_cast<int32_t>(layout.input_elements);
    for (int32_t i = tid; i < staged_inputs; i += THREADS) {
        smem_input[i] = detail::cpoly::LoadRelativeInput(input_b, input_base, input_len, i);
    }
    __syncthreads();

    scalar_t a0r[OUTPUTS_PER_THREAD]{};
    scalar_t a0i[OUTPUTS_PER_THREAD]{};
    scalar_t a1r[OUTPUTS_PER_THREAD]{};
    scalar_t a1i[OUTPUTS_PER_THREAD]{};
    for (int32_t p = 0; p < P; p++) {
        const filter_pair_t hv = smem_filter[p];
        #pragma unroll
        for (int q = 0; q < OUTPUTS_PER_THREAD; q++) {
            const int32_t local = tid + q * THREADS;
            // XOR gives the sum's LSB (1 means odd) without risking overflow.
            const bool odd_output = ((block_start ^ local ^ out_elem_offset) & 1) != 0;
            const int32_t pair_base = local + 2 * P - 2 * p;
            const shared_input_t even = smem_input[pair_base - (odd_output ? 1 : 0)];
            const shared_input_t odd = smem_input[pair_base - (odd_output ? 0 : 1)];
            const scalar_t h0 = odd_output ? hv.odd : hv.even;
            const scalar_t h1 = odd_output ? hv.even : hv.odd;
            if constexpr (ComplexInput) {
                a0r[q] = h0 * even.real() + a0r[q];
                a0i[q] = h0 * even.imag() + a0i[q];
                a1r[q] = h1 * odd.real() + a1r[q];
                a1i[q] = h1 * odd.imag() + a1i[q];
            } else {
                a0r[q] = h0 * even + a0r[q];
                a1r[q] = h1 * odd + a1r[q];
            }
        }
    }

    ChannelizePolyFusedM2Store<THREADS, OUTPUTS_PER_THREAD>(
        output_b, output_len, block_start, tid, a0r, a0i, a1r, a1i);
}

// General body for fused channel counts K*2^n, where K is a supported small
// odd radix. Keeping the filtered branch values on chip removes the global-
// memory round trip between the FIR kernel and cuFFT.
template <int THREADS, int NUM_CHAN, int NROWS, bool MaximallyDecimated,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType, typename IdxT>
__device__ __forceinline__ void ChannelizePolyFusedRadixPow2Body(
    OutType &output, const InType &input, const FilterType &filter,
    IdxT elems_per_channel_per_cta, IdxT decimation_factor, IdxT out_elem_offset, uint8_t *smem_raw)
{
    using config = detail::cpoly::FusedRadixConfig<NUM_CHAN, AccumType>;
    constexpr int RADIX_1 = config::Radix1;
    constexpr int RADIX_2 = config::Radix2;
    constexpr int ROW_STRIDE = config::RowStride;
    constexpr int FIR_OUTPUTS_PER_THREAD = config::FirOutputsPerThread;
    static_assert(RADIX_1 == 1 || RADIX_1 == 3 || RADIX_1 == 5);
    static_assert(ROW_STRIDE <= THREADS);
    static_assert(NROWS == config::NRows);
    // FFT loops execute complete warps, including padded final output rows.
    static_assert(!config::WarpFft || (THREADS % 32 == 0 && (NROWS * RADIX_2) % 32 == 0));

    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;
    static_assert(!is_complex_v<AccumType>,
        "channelize_poly: accumulator type must be real; "
        "it will be treated as complex when necessary");
    using filtering_accum_t = cuda::std::conditional_t<
        is_complex_v<input_t> || is_complex_v<filter_t>,
        typename detail::scalar_to_complex<AccumType>::ctype, AccumType>;
    using complex_accum_t = typename detail::scalar_to_complex<AccumType>::ctype;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank - 2;

    const IdxT input_len = static_cast<IdxT>(input.Size(InRank - 1));
    const IdxT output_len = static_cast<IdxT>(output.Size(OutElemRank));
    const IdxT filter_len = static_cast<IdxT>(filter.Size(0));
    const int32_t P = static_cast<int32_t>((filter_len + NUM_CHAN - 1) / NUM_CHAN);
    const detail::cpoly::FusedRadixSmemLayout<
        NUM_CHAN, MaximallyDecimated, input_t, filter_t, AccumType>
        layout(P, decimation_factor);
    const int32_t input_elements = static_cast<int32_t>(layout.input_elements);
    const int32_t height = input_elements / NUM_CHAN;

    filter_t *smem_filter = reinterpret_cast<filter_t *>(smem_raw);
    input_t *smem_input = reinterpret_cast<input_t *>(smem_raw + layout.input_offset);
    complex_accum_t *smem_work = reinterpret_cast<complex_accum_t *>(smem_raw + layout.work_offset);
    complex_accum_t *smem_stage1 =
        reinterpret_cast<complex_accum_t *>(smem_raw + layout.stage1_offset);
    complex_accum_t *twiddle_cross = reinterpret_cast<complex_accum_t *>(
        smem_raw + layout.twiddle_cross_offset);
    complex_accum_t *twiddle_radix2 = reinterpret_cast<complex_accum_t *>(
        smem_raw + layout.twiddle_radix2_offset);

    auto [input_b, output_b, filter_acc] =
        detail::cpoly::MakeAccessors<IsUnitStride>(output, input, filter, blockIdx.z);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t fir_group = tid / ROW_STRIDE;
    const int32_t channel = tid % ROW_STRIDE;
    const bool active = channel < NUM_CHAN;

    const int32_t filter_elements = static_cast<int32_t>(layout.filter_elements);
    for (int32_t i = tid; i < filter_elements; i += THREADS) {
        smem_filter[i] = (i < filter_len)
            ? filter_acc(static_cast<IdxT>(i)) : static_cast<filter_t>(0);
    }
    if constexpr (RADIX_1 > 1) {
        for (int32_t i = tid; i < RADIX_1 * RADIX_2; i += THREADS) {
            const int32_t k = i / RADIX_2;
            const int32_t n = i % RADIX_2;
            twiddle_cross[i] = ChannelizePolyTwiddle<AccumType>(k * n, NUM_CHAN);
        }
    }
    if constexpr (RADIX_2 > 1) {
        if (tid < RADIX_2 / 2) {
            const auto twiddle = ChannelizePolyRadix2Twiddle<AccumType, RADIX_2>(tid);
            // Smaller stages reuse a subset of the largest stage's roots.
            #pragma unroll
            for (int32_t len = 2; len <= RADIX_2; len *= 2) {
                const int32_t stride = RADIX_2 / len;
                if (tid % stride == 0) {
                    twiddle_radix2[len + tid / stride] = twiddle;
                }
            }
        }
    }

    const IdxT start_elem = static_cast<IdxT>(blockIdx.x) * elems_per_channel_per_cta;
    const IdxT last_elem = cuda::std::min(
        output_len - 1, start_elem + elems_per_channel_per_cta - 1);
    if constexpr (MaximallyDecimated) {
        const IdxT first_global_row = start_elem + out_elem_offset - (P - 1);
        const IdxT input_base = first_global_row * NUM_CHAN;
        for (int32_t linear = tid;
             linear < input_elements; linear += THREADS) {
            int32_t destination = linear;
            if constexpr (!config::WarpFft) {
                const int32_t row_offset = linear / NUM_CHAN;
                const int32_t col = linear % NUM_CHAN;
                const IdxT global_row = first_global_row + row_offset;
                int32_t smem_row = static_cast<int32_t>(global_row % height);
                if (smem_row < 0) smem_row += height;
                destination = smem_row * NUM_CHAN + col;
            }
            smem_input[destination] = detail::cpoly::LoadRelativeInput(
                input_b, input_base, input_len, linear);
        }
    }
    __syncthreads();

    const IdxT last_start = start_elem + ((last_elem - start_elem) / NROWS) * NROWS;
    int32_t newest_row = P + (fir_group + 1) * FIR_OUTPUTS_PER_THREAD - 2;
    int32_t reload_row = 0;
    for (IdxT next_start = start_elem;
         next_start <= last_start; next_start += NROWS) {
        IdxT first_raw = 0;
        if constexpr (!MaximallyDecimated) {
            const IdxT input_base = (next_start + out_elem_offset) * decimation_factor;
            const int32_t first_offset = static_cast<int32_t>(decimation_factor) - 1 - P * NUM_CHAN;
            first_raw = input_base + first_offset;
            for (int32_t i = tid; i < input_elements; i += THREADS) {
                smem_input[i] = detail::cpoly::LoadRelativeInput(input_b, first_raw, input_len, i);
            }
            __syncthreads();
        }

        filtering_accum_t filtered[FIR_OUTPUTS_PER_THREAD]{};
        bool valid[FIR_OUTPUTS_PER_THREAD];
        #pragma unroll
        for (int q = 0; q < FIR_OUTPUTS_PER_THREAD; q++) {
            const IdxT t = next_start + fir_group * FIR_OUTPUTS_PER_THREAD + q;
            valid[q] = active && t <= last_elem;
        }

        if constexpr (MaximallyDecimated) {
            if (active) {
                int32_t sample_row = newest_row;
                if constexpr (!config::WarpFft) {
                    const IdxT first_t = next_start + fir_group * FIR_OUTPUTS_PER_THREAD;
                    sample_row = static_cast<int32_t>(
                        (first_t + out_elem_offset + FIR_OUTPUTS_PER_THREAD - 1) % height);
                }
                for (int32_t r = 0;
                     r < P + FIR_OUTPUTS_PER_THREAD - 1; r++) {
                    const input_t iv = smem_input[sample_row * NUM_CHAN + (NUM_CHAN - 1 - channel)];
                    const auto iav = detail::channelize_cast_operand<filtering_accum_t>(iv);
                    #pragma unroll
                    for (int q = 0; q < FIR_OUTPUTS_PER_THREAD; q++) {
                        const int32_t p = q + r - (FIR_OUTPUTS_PER_THREAD - 1);
                        if (valid[q] && p >= 0 && p < P) {
                            detail::channelize_cmac(
                                filtered[q], detail::channelize_cast_operand<filtering_accum_t>(
                                    smem_filter[p * NUM_CHAN + channel]),
                                iav);
                        }
                    }
                    if (--sample_row < 0) sample_row += height;
                }
            }
        } else if (active) {
            #pragma unroll
            for (int q = 0; q < FIR_OUTPUTS_PER_THREAD; q++) {
                if (valid[q]) {
                    const IdxT t = next_start +
                        fir_group * FIR_OUTPUTS_PER_THREAD + q + out_elem_offset;
                    const IdxT last_arrived = t * decimation_factor + decimation_factor - 1;
                    const int32_t remapped =
                        (channel + NUM_CHAN -
                         static_cast<int32_t>(decimation_factor)) % NUM_CHAN;
                    const int32_t branch = NUM_CHAN - 1 - remapped;
                    if (last_arrived >= branch) {
                        const IdxT delta = last_arrived - branch;
                        const IdxT newest = last_arrived - delta % NUM_CHAN;
                        const int32_t phase = static_cast<int32_t>(
                            (channel + t * decimation_factor) % NUM_CHAN);
                        for (int32_t p = 0; p < P; p++) {
                            const IdxT sample = newest - static_cast<IdxT>(p) * NUM_CHAN;
                            const input_t iv = smem_input[sample - first_raw];
                            detail::channelize_cmac(
                                filtered[q], detail::channelize_cast_operand<filtering_accum_t>(
                                    smem_filter[p * NUM_CHAN + phase]),
                                detail::channelize_cast_operand<filtering_accum_t>(iv));
                        }
                    }
                }
            }
        }

        if (active) {
            int32_t fft_channel = channel;
            if constexpr (RADIX_1 == 1) {
                // Store pure-power-of-two FIR branches in FFT input order.
                // No radix-K stage or cross twiddle is needed in this case.
                fft_channel = detail::cpoly::BitReverse<RADIX_2>(channel);
            }
            #pragma unroll
            for (int q = 0; q < FIR_OUTPUTS_PER_THREAD; q++) {
                const int32_t row = fir_group * FIR_OUTPUTS_PER_THREAD + q;
                if constexpr (is_complex_v<filtering_accum_t>) {
                    smem_work[row * NUM_CHAN + fft_channel] =
                        static_cast<complex_accum_t>(filtered[q]);
                } else {
                    smem_work[row * NUM_CHAN + fft_channel] = {
                        static_cast<AccumType>(filtered[q]), static_cast<AccumType>(0)};
                }
            }
        }
        __syncthreads();
        if constexpr (config::WarpFft) {
            for (int32_t logical = tid;
                 logical < NROWS * RADIX_2; logical += THREADS) {
                const int32_t row = logical / RADIX_2;
                const int32_t lane = logical % RADIX_2;
                const int32_t bit_reversed = detail::cpoly::BitReverse<RADIX_2>(lane);
                complex_accum_t value[RADIX_1];
                if constexpr (RADIX_1 == 1) {
                    // Pure-power-of-two FIR stores are already bit reversed.
                    value[0] = smem_work[row * NUM_CHAN + lane];
                } else {
                    complex_accum_t branch[RADIX_1];
                    #pragma unroll
                    for (int k = 0; k < RADIX_1; ++k) {
                        branch[k] = smem_work[row * NUM_CHAN + bit_reversed + k * RADIX_2];
                    }
                    ChannelizePolySmallDFT(branch, value);
                    #pragma unroll
                    for (int k = 1; k < RADIX_1; ++k) {
                        complex_accum_t twiddled{};
                        detail::channelize_cmac(twiddled,
                            twiddle_cross[k * RADIX_2 + bit_reversed], value[k]);
                        value[k] = twiddled;
                    }
                }
                constexpr unsigned mask = 0xffffffffU;
                #pragma unroll
                for (int len = 2; len <= RADIX_2; len *= 2) {
                    const int j = lane % (len / 2);
                    const bool lower = (lane & (len / 2)) == 0;
                    const complex_accum_t twiddle = twiddle_radix2[len + j];
                    #pragma unroll
                    for (int k = 0; k < RADIX_1; ++k) {
                        const complex_accum_t partner{
                            __shfl_xor_sync(mask, value[k].real(), len / 2, RADIX_2),
                            __shfl_xor_sync(mask, value[k].imag(), len / 2, RADIX_2)};
                        const complex_accum_t lo = lower ? value[k] : partner;
                        const complex_accum_t hi = lower ? partner : value[k];
                        complex_accum_t product{};
                        detail::channelize_cmac(product, twiddle, hi);
                        value[k] = lower ? lo + product : lo - product;
                    }
                }
                if constexpr (RADIX_1 > 1) {
                    // Warp shuffles do not order shared loads before reuse.
                    __syncwarp(mask);
                }
                #pragma unroll
                for (int k = 0; k < RADIX_1; ++k) {
                    if constexpr (RADIX_1 == 1) {
                        const IdxT t = next_start + row;
                        if (t <= last_elem) {
                            output_b(t, static_cast<IdxT>(lane)) = static_cast<output_t>(value[k]);
                        }
                    } else {
                        // Transpose the register-owned odd-radix outputs for
                        // coalesced global stores across all CTA threads.
                        smem_work[row * NUM_CHAN + lane * RADIX_1 + k] = value[k];
                    }
                }
            }
            if constexpr (RADIX_1 > 1) {
                __syncthreads();
            }
        } else {
            if constexpr (RADIX_1 > 1) {
                for (int32_t logical = tid;
                     logical < NROWS * RADIX_2; logical += THREADS) {
                    const int32_t row = logical / RADIX_2;
                    const int32_t n1 = logical % RADIX_2;
                    const int32_t base = row * NUM_CHAN + n1;
                    const int32_t stage_base = row * NUM_CHAN + n1 * RADIX_1;
                    // Array store loops increased register/stack usage on L4.
                    if constexpr (RADIX_1 == 3) {
                        const complex_accum_t x0 = smem_work[base];
                        const complex_accum_t x1 = smem_work[base + RADIX_2];
                        const complex_accum_t x2 = smem_work[base + 2 * RADIX_2];
                        complex_accum_t y0, y1, y2;
                        ChannelizePolyRadix3(x0, x1, x2, y0, y1, y2);
                        smem_stage1[stage_base] = y0;
                        smem_stage1[stage_base + 1] = y1;
                        smem_stage1[stage_base + 2] = y2;
                    } else {
                        const complex_accum_t x0 = smem_work[base];
                        const complex_accum_t x1 = smem_work[base + RADIX_2];
                        const complex_accum_t x2 = smem_work[base + 2 * RADIX_2];
                        const complex_accum_t x3 = smem_work[base + 3 * RADIX_2];
                        const complex_accum_t x4 = smem_work[base + 4 * RADIX_2];
                        complex_accum_t y0, y1, y2, y3, y4;
                        ChannelizePolyRadix5(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4);
                        smem_stage1[stage_base] = y0;
                        smem_stage1[stage_base + 1] = y1;
                        smem_stage1[stage_base + 2] = y2;
                        smem_stage1[stage_base + 3] = y3;
                        smem_stage1[stage_base + 4] = y4;
                    }
                }
                __syncthreads();

                for (int32_t logical = tid;
                     logical < NROWS * NUM_CHAN; logical += THREADS) {
                    const int32_t row = logical / NUM_CHAN;
                    const int32_t fft_channel = logical % NUM_CHAN;
                    const int32_t n1 = fft_channel / RADIX_1;
                    const int32_t k2 = fft_channel % RADIX_1;
                    const int32_t bit_reversed = detail::cpoly::BitReverse<RADIX_2>(n1);
                    complex_accum_t value{};
                    detail::channelize_cmac(
                        value, twiddle_cross[k2 * RADIX_2 + n1],
                        smem_stage1[row * NUM_CHAN + n1 * RADIX_1 + k2]);
                    smem_work[row * NUM_CHAN + bit_reversed * RADIX_1 + k2] = value;
                }
                __syncthreads();
            }

            for (int32_t len = 2; len <= RADIX_2; len *= 2) {
                for (int32_t logical = tid;
                     logical < NROWS * NUM_CHAN; logical += THREADS) {
                    const int32_t row = logical / NUM_CHAN;
                    const int32_t fft_channel = logical % NUM_CHAN;
                    const int32_t k1 = fft_channel / RADIX_1;
                    const int32_t k2 = fft_channel % RADIX_1;
                    const int32_t j = k1 % len;
                    if (j < len / 2) {
                        const int32_t base = k1 - j;
                        const int32_t lo = row * NUM_CHAN + (base + j) * RADIX_1 + k2;
                        const int32_t hi = lo + (len / 2) * RADIX_1;
                        const complex_accum_t u = smem_work[lo];
                        complex_accum_t v{};
                        detail::channelize_cmac(v, twiddle_radix2[len + j], smem_work[hi]);
                        smem_work[lo] = u + v;
                        smem_work[hi] = u - v;
                    }
                }
                __syncthreads();
            }
        }
        if constexpr (!config::WarpFft || RADIX_1 > 1) {
            for (int32_t logical = tid;
                 logical < NROWS * NUM_CHAN; logical += THREADS) {
                const int32_t row = logical / NUM_CHAN;
                const int32_t fft_channel = logical % NUM_CHAN;
                const IdxT t = next_start + row;
                if (t <= last_elem) {
                    output_b(t, static_cast<IdxT>(fft_channel)) =
                        static_cast<output_t>(smem_work[logical]);
                }
            }
        }
        __syncthreads();

        if constexpr (MaximallyDecimated) {
          if (next_start < last_start) {
            const IdxT first_global_row = next_start + out_elem_offset + NROWS;
            const IdxT input_base = first_global_row * NUM_CHAN;
            for (int32_t logical = tid;
                 logical < NROWS * NUM_CHAN; logical += THREADS) {
                const int32_t row = logical / NUM_CHAN;
                const int32_t col = logical % NUM_CHAN;
                int32_t smem_row = reload_row + row;
                if constexpr (config::WarpFft) {
                    if (smem_row >= height) smem_row -= height;
                } else {
                    const IdxT global_row = first_global_row + row;
                    smem_row = static_cast<int32_t>(global_row % height);
                }
                smem_input[smem_row * NUM_CHAN + col] =
                    detail::cpoly::LoadRelativeInput(
                        input_b, input_base, input_len, logical);
            }
            if constexpr (config::WarpFft) {
                newest_row += NROWS;
                if (newest_row >= height) newest_row -= height;
                reload_row += NROWS;
                if (reload_row >= height) reload_row -= height;
            }
            __syncthreads();
          }
        }
    }
}

// One source-level kernel family covers row-owned small channel counts, pure
// powers of two, and supported K*2^n channel counts. Runtime dispatch
// deliberately instantiates only a small supported set of NUM_CHAN values.
template <int THREADS, int NUM_CHAN, int NROWS, bool MaximallyDecimated,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType, typename IdxT>
__device__ __forceinline__ void ChannelizePolyFusedRadixBody(
    OutType &output, const InType &input, const FilterType &filter,
    IdxT elems_per_channel_per_cta, IdxT decimation_factor, IdxT out_elem_offset)
{
    extern __shared__ __align__(16) uint8_t smem_raw[];
    using config = detail::cpoly::FusedRadixConfig<
        NUM_CHAN, AccumType, MaximallyDecimated, typename InType::value_type>;
    static_assert(THREADS == config::Threads);
    constexpr int OUTPUTS_PER_THREAD = config::SmallOutputsPerThread;
    if constexpr (NUM_CHAN <= 6 && MaximallyDecimated) {
        using traits = ChannelizePolyFusedSmallTraits<
            NUM_CHAN, OutType, InType, FilterType, AccumType>;
        if constexpr (NUM_CHAN == 2 && traits::UseM2PairLeaf) {
            ChannelizePolyFusedM2D2Body<
                THREADS, OUTPUTS_PER_THREAD, IsUnitStride, OutType, InType, FilterType, AccumType>(
                    output, input, filter, out_elem_offset, elems_per_channel_per_cta, smem_raw);
        } else {
            const IdxT P = (filter.Size(FilterType::Rank() - 1) + NUM_CHAN - 1) / NUM_CHAN;
            if (detail::cpoly::FusedSmallUseDirect<
                    NUM_CHAN, OutType, InType, FilterType, AccumType>(
                        P, output.Size(OutType::Rank() - 2))) {
                ChannelizePolyFusedSmallDirectBody<
                    THREADS, NUM_CHAN, OUTPUTS_PER_THREAD, IsUnitStride,
                    OutType, InType, FilterType, AccumType>(
                        output, input, filter, out_elem_offset);
            } else {
                ChannelizePolyFusedSmallCachedBody<
                    THREADS, NUM_CHAN, OUTPUTS_PER_THREAD, IsUnitStride,
                    OutType, InType, FilterType, AccumType>(
                        output, input, filter, out_elem_offset, smem_raw);
            }
        }
    } else if constexpr (NUM_CHAN == 2) {
        ChannelizePolyFusedM2D1Body<
            THREADS, OUTPUTS_PER_THREAD, IsUnitStride, OutType, InType, FilterType, AccumType>(
                output, input, filter, out_elem_offset, smem_raw);
    } else {
        ChannelizePolyFusedRadixPow2Body<
            THREADS, NUM_CHAN, NROWS, MaximallyDecimated, IsUnitStride,
            OutType, InType, FilterType, AccumType>(
                output, input, filter, elems_per_channel_per_cta,
                decimation_factor, out_elem_offset, smem_raw);
    }
}

// Mutually exclusive overloads omit the optional bound instead of passing zero.
// Select index width explicitly, not from the types of integer launch literals.
template <int THREADS, int NUM_CHAN, int NROWS, bool MaximallyDecimated,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType, typename IdxT = index_t,
          int MIN_BLOCKS = sizeof(IdxT) <= sizeof(int32_t) ?
              detail::cpoly::FusedRadixConfig<NUM_CHAN, AccumType,
                  MaximallyDecimated, typename InType::value_type>::MinBlocksPerSm : 0>
    requires (MIN_BLOCKS == 0)
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D_FusedRadixPow2(
    OutType output, InType input, FilterType filter,
    cuda::std::type_identity_t<IdxT> elems_per_channel_per_cta,
    cuda::std::type_identity_t<IdxT> decimation_factor,
    cuda::std::type_identity_t<IdxT> out_elem_offset)
{
    ChannelizePolyFusedRadixBody<THREADS, NUM_CHAN, NROWS, MaximallyDecimated,
        IsUnitStride, OutType, InType, FilterType, AccumType>(
            output, input, filter, elems_per_channel_per_cta, decimation_factor, out_elem_offset);
}

template <int THREADS, int NUM_CHAN, int NROWS, bool MaximallyDecimated,
          bool IsUnitStride, typename OutType, typename InType,
          typename FilterType, typename AccumType, typename IdxT = index_t,
          int MIN_BLOCKS = sizeof(IdxT) <= sizeof(int32_t) ?
              detail::cpoly::FusedRadixConfig<NUM_CHAN, AccumType,
                  MaximallyDecimated, typename InType::value_type>::MinBlocksPerSm : 0>
    requires (MIN_BLOCKS > 0)
__launch_bounds__(THREADS, MIN_BLOCKS)
__global__ void ChannelizePoly1D_FusedRadixPow2(
    OutType output, InType input, FilterType filter,
    cuda::std::type_identity_t<IdxT> elems_per_channel_per_cta,
    cuda::std::type_identity_t<IdxT> decimation_factor,
    cuda::std::type_identity_t<IdxT> out_elem_offset)
{
    ChannelizePolyFusedRadixBody<THREADS, NUM_CHAN, NROWS, MaximallyDecimated,
        IsUnitStride, OutType, InType, FilterType, AccumType>(
            output, input, filter, elems_per_channel_per_cta, decimation_factor, out_elem_offset);
}

// Unpack the compressed representation of the spectrum after a real-to-complex FFT.
// Because the input was real, the spectrum is conjugate symmetric and fft() will
// return a packed version of the output that includes only the unique elements
// (up to conjugate symmetry). We unpack because for the channelizer we want all
// channel outputs.
template <bool IsUnitStride, typename DataType>
__global__ void ChannelizePoly1DUnpackDFT(DataType inout)
{
    constexpr int Rank = DataType::Rank();
    constexpr int ChannelRank = Rank-1;
    constexpr int ElemRank = Rank-2;
    using value_t = typename DataType::value_type;

    const index_t num_elem_per_channel = inout.Size(ElemRank);
    const index_t num_channels = inout.Size(ChannelRank);

    const index_t mid = num_channels/2 + 1;

    // Bind batch coords; remaining dims are (elem, channel). Access with
    // inout_b(elem, chan).
    detail::TensorAccessor<DataType, IsUnitStride> inout_acc(inout);
    const auto batch_idx = BlockToIdx(inout, blockIdx.x, 2);
    auto inout_b = detail::bind_first_n<Rank - 2>(inout_acc, batch_idx);

    const index_t upper = (num_channels % 2 == 0) ? (mid - 1) : mid;
    // The launch caps grid.y at the grid.y limit; stride over the remainder.
    const index_t stride = static_cast<index_t>(gridDim.y) * blockDim.x;
    for (index_t elem = static_cast<index_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         elem < num_elem_per_channel; elem += stride) {
        for (index_t i = 1; i < upper; i++) {
            const value_t val = inout_b(elem, i);
            inout_b(elem, i) = conj(val);
            inout_b(elem, num_channels - i) = val;
        }
    }
}

#endif // __CUDACC__

}; // namespace matx
