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

namespace matx {

// detail constants that require both host and device visibility at compile
// time. Scoped to matx::detail::cpoly so the transform's helpers can sit in
// the same namespace without colliding with other transforms' internals.
namespace detail {
namespace cpoly {
    // Number of output elements generated per thread
    constexpr index_t ElemsPerThread = 1;

    // Maximum number of filter rotations per channel for the SmemTiled kernel.
    // This is used to determine if the filter can be stored in shared memory.
    // The number of rotations can exceed this value, but the filter will be
    // read from global memory rather than cached in shared memory.
    constexpr int SmemTiledMaxRotations = 32;
} // namespace cpoly
} // namespace detail

#ifdef __CUDACC__ 

namespace detail {

template <typename AccumT, typename FilterT>
__MATX_DEVICE__ __MATX_INLINE__ auto channelize_cast_filter(FilterT v)
{
    if constexpr (is_complex_v<FilterT>) {
        // Complex filter: keep full complex multiply
        return static_cast<AccumT>(v);
    } else if constexpr (is_complex_v<AccumT>) {
        // Real filter + complex accumulator: promote to scalar only
        using accum_scalar_t = typename inner_op_type_t<AccumT>::type;
        return static_cast<accum_scalar_t>(v);
    } else {
        return static_cast<AccumT>(v);
    }
}

template <typename AccumT, typename InputT>
__MATX_DEVICE__ __MATX_INLINE__ auto channelize_cast_input(InputT v)
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
// Fused complex multiply-accumulate. Decomposes the operation into scalar
// FMA instructions so the compiler emits 4 FFMA per complex tap instead of
// ~8 mixed FMUL/FADD/FSUB. Falls back to the default operator* + operator+=
// for real or mixed-precision types.
template <typename AccumT, typename FilterValT, typename InputValT>
__MATX_DEVICE__ __MATX_INLINE__ void channelize_cmac(
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

template <int THREADS, bool MaximallyDecimated, bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename AccumType>
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D(OutType output, InType input, FilterType filter, index_t decimation_factor, uint32_t smem_filter_bytes)
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

    const int elem_block = blockIdx.x;
    const int channel = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr index_t ELEMS_PER_BLOCK = detail::cpoly::ElemsPerThread * THREADS;
    const index_t first_out_elem = elem_block * detail::cpoly::ElemsPerThread * THREADS;
    const index_t last_out_elem = cuda::std::min(
        output_len_per_channel - 1, first_out_elem + ELEMS_PER_BLOCK - 1);

    // Wrap input/output/filter in TensorAccessor and bind the per-block batch
    // coords once. After binding, per-access calls supply only the inner
    // indices: (sample_idx) for input, (t, channel) for output. On the fast
    // path this collapses to base_ptr[stride*inner + ...] arithmetic with no
    // per-access stride reload; on the slow path it forwards to operator().
    //
    // Note on output layout: MatX arranges output as [batch..., elem, channel]
    // where channel is the LAST dim (see the size asserts in
    // channelize_poly_impl). We therefore bind only the batch dims (first
    // OutRank-2) and pass both elem (t) and channel at access time.
    detail::TensorAccessor<InType, IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType, IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);

    const auto in_batch_idx = BlockToIdx(input, blockIdx.z, 1);   // last slot unused
    const auto out_batch_idx = BlockToIdx(output, blockIdx.z, 2); // last two slots unused

    auto input_b  = detail::bind_first_n<InRank  - 1>(input_acc,  in_batch_idx);
    auto output_b = detail::bind_first_n<OutRank - 2>(output_acc, out_batch_idx);

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
                accum_t accum {};
                index_t sample_idx = s + t * num_channels;
                index_t h_skip = 0;
                if (sample_idx >= input_len) {
                    h_skip = 1;
                    sample_idx -= num_channels;
                }
                const filter_t *h = smem_filter + h_skip;
                int niter = static_cast<int>(cuda::std::min(filter_phase_len - h_skip, t + 1 - h_skip));
                for (int i = 0; i < niter; i++) {
                    const input_t in_val = input_b(sample_idx);
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_filter<accum_t>(*h),
                            detail::channelize_cast_input<accum_t>(in_val));
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
                accum_t accum {};
                index_t sample_idx = s + t * num_channels;
                index_t h_skip = 0;
                if (sample_idx >= input_len) {
                    h_skip = 1;
                    sample_idx -= num_channels;
                }
                index_t h_ind = channel + h_skip * num_channels;
                index_t niter = cuda::std::min(available_taps - h_skip, t + 1 - h_skip);
                for (index_t i = 0; i < niter; i++) {
                    const input_t in_val = input_b(sample_idx);
                    const filter_t h_val = filter_acc(h_ind);
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_filter<accum_t>(h_val),
                            detail::channelize_cast_input<accum_t>(in_val));
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
            const index_t last_arrived = t * decimation_factor + decimation_factor - 1;
            index_t niter = 0;
            const index_t phase = (channel + t * decimation_factor) % num_channels;
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
                        detail::channelize_cast_filter<accum_t>(h_val),
                        detail::channelize_cast_input<accum_t>(in_val));
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
// Block: dim3(CTILE, NOUT)
// Grid:  dim3(time_blocks, channel_tiles, batches)
//
// Shared memory layout (FilterInSmem = true):
//   smem_filter: [P][CTILE] for D==M, or [CTILE][K][P] for D<M
//   smem_input:  [height][CTILE] circular buffer, height = P + NOUT - 1
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
__launch_bounds__(CTILE * NOUT)
__global__ void ChannelizePoly1D_SmemTiled(
    OutType output, InType input, FilterType filter,
    IdxT elems_per_channel_per_cta, IdxT decimation_factor, int32_t num_phases_per_channel)
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
    assert(blockDim.x * blockDim.y == CTILE * NOUT && blockDim.z == 1);
    const int32_t nthreads = CTILE * NOUT;
    const int32_t tile_base = static_cast<int32_t>(blockIdx.y) * CTILE;
    const int32_t c = tile_base + cx;
    const bool active = (c < M);

    // Branch remap offset for Harris convention (oversampled only; 0 for D == M).
    const int32_t L = MaximallyDecimated ? 0 : (M - static_cast<int32_t>(decimation_factor));

    const int32_t filter_stride = K * P; // per-channel filter block size
    const int32_t height = P + NOUT - 1;

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

    // TensorAccessors bind per-block batch coords once. After binding,
    // input_b(sample_idx) reads one input sample for this batch, and
    // output_b(t, ch) writes one output sample. Fast path folds the strides
    // into pointer arithmetic; slow path forwards to operator().
    detail::TensorAccessor<InType, IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType, IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);

    const auto in_batch_idx  = BlockToIdx(input,  blockIdx.z, 1);
    const auto out_batch_idx = BlockToIdx(output, blockIdx.z, 2);
    auto input_b  = detail::bind_first_n<InRank  - 1>(input_acc,  in_batch_idx);
    auto output_b = detail::bind_first_n<OutRank - 2>(output_acc, out_batch_idx);

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
            // TODO-PERF: Each channel rotates through K filter phases. Currently, we store all K phases
            // per channel in shared memory. It is possible for K * CTILE to exceed the total number
            // of channels, in which case we would be better off storing the full filter in shared memory.
            // Furthermore, if there are fewer than K output points per channel generated in this CTA,
            // then we could store only the required phases.
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

    auto max_bidx = [&](IdxT t) -> IdxT {
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
        // bidx = t, causal_count = t+1, last_arrived >= s always true.
        // Track buf_row incrementally across iterations to avoid modulo.

        // Seed buf_row for ty=0 at start_elem
        int32_t buf_row_base = static_cast<int32_t>(start_elem % height);
        // Per-iteration advance: NOUT output steps = NOUT buf_row advance. This is a defensive modulo
        // so that we can keep buf_row_base in [0, height) with only a conditional subtraction.
        const int32_t nout_wrap = NOUT % height;

        const IdxT last_start = start_elem + ((last_elem - start_elem) / NOUT) * NOUT;
        for (IdxT next_start = start_elem; next_start <= last_start; next_start += NOUT) {
            const IdxT t = next_start + ty;
            if (t <= last_elem && active) {
                accum_t accum{};

                // buf_row for this thread's output step
                int32_t my_buf_row = buf_row_base + ty;
                if (my_buf_row >= height) my_buf_row -= height;

                const IdxT newest_raw = static_cast<IdxT>(s) + t * M;
                int32_t h_skip = 0;
                int32_t niter = static_cast<int32_t>(
                    cuda::std::min(static_cast<IdxT>(P), t + 1));
                if (newest_raw >= input_len) {
                    h_skip = 1;
                    niter = static_cast<int32_t>(
                        cuda::std::min(static_cast<IdxT>(P - 1), t));
                    if (--my_buf_row < 0) my_buf_row += height;
                }

                const int32_t prologue = cuda::std::min(my_buf_row + 1, niter);
                const int32_t epilogue = niter - prologue;
                // Single running counter instead of separate `p` and `h_ind`.
                // Each layout's access pattern reduces to (init, stride):
                //   Full:    smem[(p+h_skip)*M + c]           -> init h_skip*M+c, stride M
                //   Rotated: smem[(p+h_skip)*CTILE + cx]      -> init h_skip*CTILE+cx, stride CTILE
                //   Global:  filter[c + (p+h_skip)*M]         -> init c+h_skip*M, stride M
                int32_t filter_idx;
                if constexpr (FilterInSmem && !FilterFullLayout) {
                    filter_idx = h_skip * CTILE + cx;
                } else {
                    filter_idx = c + h_skip * M;
                }
                for (int32_t i = 0; i < prologue; i++) {
                    filter_t hv;
                    if constexpr (FilterInSmem) {
                        hv = smem_filter_base[filter_idx];
                    } else {
                        hv = filter_acc(static_cast<index_t>(filter_idx));
                    }
                    const input_t iv = smem_input[my_buf_row * CTILE + cx];
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_filter<accum_t>(hv),
                            detail::channelize_cast_input<accum_t>(iv));
                    my_buf_row--;
                    if constexpr (FilterInSmem && !FilterFullLayout) {
                        filter_idx += CTILE;
                    } else {
                        filter_idx += M;
                    }
                }
                my_buf_row = height - 1;
                for (int32_t i = 0; i < epilogue; i++) {
                    filter_t hv;
                    if constexpr (FilterInSmem) {
                        hv = smem_filter_base[filter_idx];
                    } else {
                        hv = filter_acc(static_cast<index_t>(filter_idx));
                    }
                    const input_t iv = smem_input[my_buf_row * CTILE + cx];
                    detail::channelize_cmac(accum,
                            detail::channelize_cast_filter<accum_t>(hv),
                            detail::channelize_cast_input<accum_t>(iv));
                    my_buf_row--;
                    if constexpr (FilterInSmem && !FilterFullLayout) {
                        filter_idx += CTILE;
                    } else {
                        filter_idx += M;
                    }
                }

                output_b(t, c) = static_cast<output_t>(accum);
            }

            if (next_start < last_start) {
                // Incremental buf_row_base advance (no modulo)
                buf_row_base += nout_wrap;
                if (buf_row_base >= height) buf_row_base -= height;

                // Ensure all threads have finished reading smem_input before overwriting
                __syncthreads();

                // Load NOUT new rows. For D==M, exactly NOUT new samples arrive.
                // Each ty-lane loads one row (its cx column).
                {
                    const IdxT next_end = cuda::std::min(next_start + static_cast<IdxT>(2 * NOUT) - 1, last_elem);
                    const int32_t new_rows = static_cast<int32_t>(max_bidx(next_end) - loaded_up_to);
                    int32_t lr = static_cast<int32_t>((loaded_up_to + 1) % height);
                    if (ty < new_rows) {
                        int32_t my_lr = lr + ty;
                        if (my_lr >= height) my_lr -= height;
                        load_smem_elem(my_lr, cx, loaded_up_to + 1 + ty);
                    }
                    loaded_up_to += new_rows;
                }

                // Ensure new rows are visible before next iteration's compute
                __syncthreads();
            }
        }
    } else {
        // D < M: oversampled path
        const IdxT last_start = start_elem + ((last_elem - start_elem) / NOUT) * NOUT;
        for (IdxT next_start = start_elem; next_start <= last_start; next_start += NOUT) {
            const IdxT t = next_start + ty;
            if (t <= last_elem && active) {
                accum_t accum{};

                const IdxT last_arrived = t * decimation_factor + decimation_factor - 1;
                if (last_arrived >= s) {
                    const IdxT A = last_arrived - s;
                    const IdxT bidx = A / M;
                    const IdxT causal_count = bidx + 1;

                    const int32_t k = static_cast<int32_t>(t % K);
                    // Phase uses the original logical channel (not remapped)
                    const int32_t phase = static_cast<int32_t>(
                        (c + t * decimation_factor) % M);

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
                    for (int32_t i = 0; i < prologue; i++) {
                        filter_t hv;
                        if constexpr (FilterInSmem) {
                            hv = smem_filter_base[filter_idx];
                        } else {
                            hv = filter_acc(static_cast<index_t>(filter_idx));
                        }
                        const input_t iv = smem_input[buf_row * CTILE + cx];
                        detail::channelize_cmac(accum,
                                detail::channelize_cast_filter<accum_t>(hv),
                                detail::channelize_cast_input<accum_t>(iv));
                        buf_row--;
                        if constexpr (FilterInSmem && !FilterFullLayout) {
                            filter_idx += 1;
                        } else {
                            filter_idx += M;
                        }
                    }
                    buf_row = height - 1;
                    for (int32_t i = 0; i < epilogue; i++) {
                        filter_t hv;
                        if constexpr (FilterInSmem) {
                            hv = smem_filter_base[filter_idx];
                        } else {
                            hv = filter_acc(static_cast<index_t>(filter_idx));
                        }
                        const input_t iv = smem_input[buf_row * CTILE + cx];
                        detail::channelize_cmac(accum,
                                detail::channelize_cast_filter<accum_t>(hv),
                                detail::channelize_cast_input<accum_t>(iv));
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
template <bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename AccumType>
__global__ void ChannelizePoly1D_Smem(OutType output, InType input, FilterType filter, index_t elems_per_channel_per_cta)
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

    // TensorAccessors with per-block batch binding (see ChannelizePoly1D_SmemTiled).
    detail::TensorAccessor<InType, IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType, IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);
    const auto in_batch_idx  = BlockToIdx(input,  blockIdx.z, 1);
    const auto out_batch_idx = BlockToIdx(output, blockIdx.z, 2);
    auto input_b  = detail::bind_first_n<InRank  - 1>(input_acc,  in_batch_idx);
    auto output_b = detail::bind_first_n<OutRank - 2>(output_acc, out_batch_idx);

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
        const index_t out_sample_ind = start_elem - (filter_phase_len-1) + t;
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
            const index_t input_ind = (next_start_elem + ty) * num_channels + chan;
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
                        detail::channelize_cast_filter<accum_t>(*h),
                        detail::channelize_cast_input<accum_t>(*sample));
                sample += num_channels;
                h -= num_channels;
            }
            sample = smem_input + (num_channels - 1 - chan);
            for (int32_t k = 0; k < epilogue_count; k++) {
                detail::channelize_cmac(accum,
                        detail::channelize_cast_filter<accum_t>(*h),
                        detail::channelize_cast_input<accum_t>(*sample));
                sample += num_channels;
                h -= num_channels;
            }

            output_b(out_elem_idx, chan) = static_cast<output_t>(accum);
        }

        next_start_elem += out_samples_this_iter;
    }
}

template <int THREADS, int NUM_CHAN, bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename AccumType>
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D_FusedChan(OutType output, InType input, FilterType filter)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;
    static_assert(! is_complex_v<AccumType>,
        "channelize_poly: accumulator type must be real; it will be treated as complex when necessary");
    // If the output is complex, then then accumulator is complex. Otherwise, the accumulator is real.
    using filtering_accum_t = cuda::std::conditional_t<is_complex_v<input_t> || is_complex_v<filter_t>,
        typename detail::scalar_to_complex<AccumType>::ctype, AccumType>;
    using complex_accum_t = typename detail::scalar_to_complex<AccumType>::ctype;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int OutElemRank = OutRank-2;

    const index_t input_len = input.Size(InRank-1);
    const index_t output_len_per_channel = output.Size(OutElemRank);
    const index_t filter_full_len = filter.Size(0);
    const index_t filter_phase_len = (filter_full_len + NUM_CHAN - 1) / NUM_CHAN;

    const int elem_block = blockIdx.x;
    const int tid = threadIdx.x;

    // TensorAccessors + batch bind. Batch coords come from BlockToIdx; inner
    // indices (sample for input, (t, chan) for output) are supplied per access.
    detail::TensorAccessor<InType, IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType, IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);
    const auto in_batch_idx  = BlockToIdx(input,  blockIdx.z, 1);
    const auto out_batch_idx = BlockToIdx(output, blockIdx.z, 2);
    auto input_b  = detail::bind_first_n<InRank  - 1>(input_acc,  in_batch_idx);
    auto output_b = detail::bind_first_n<OutRank - 2>(output_acc, out_batch_idx);

    constexpr index_t ELEMS_PER_BLOCK = detail::cpoly::ElemsPerThread * THREADS;
    const index_t first_out_elem = elem_block * detail::cpoly::ElemsPerThread * THREADS;
    const index_t last_out_elem = cuda::std::min(
        output_len_per_channel - 1, first_out_elem + ELEMS_PER_BLOCK - 1);

    // Versions of CUDA prior to 11.8 do not allow static shared memory allocations of
    // cuda::std::complex types due to it having no trivial constructor. This workaround
    // prevents an 'initializer not allowed for __shared__ variable' error.
    __shared__ __align__(16) uint8_t smem_eij_workaround[sizeof(complex_accum_t)*NUM_CHAN*NUM_CHAN];
    complex_accum_t (&smem_eij)[NUM_CHAN][NUM_CHAN] = reinterpret_cast<complex_accum_t (&)[NUM_CHAN][NUM_CHAN]>(smem_eij_workaround);
    // Pre-compute the DFT complex exponentials and store in shared memory
    for (int t = tid; t < NUM_CHAN*NUM_CHAN; t += THREADS) {
        const int i = t / NUM_CHAN;
        const int j = t % NUM_CHAN;
        if constexpr (std::is_same_v<AccumType, double>) {
            const double arg = 2.0 * M_PI * j * i / NUM_CHAN;
            double sinx, cosx;
            sincos(arg, &sinx, &cosx);
            complex_accum_t eij { static_cast<AccumType>(cosx), static_cast<AccumType>(sinx) };
            smem_eij[i][j] = eij;
        } else {
            const float arg = 2.0f * static_cast<float>(M_PI) * j * i / NUM_CHAN;
            float sinx, cosx;
            sincosf(arg, &sinx, &cosx);
            complex_accum_t eij { static_cast<AccumType>(cosx), static_cast<AccumType>(sinx) };
            smem_eij[i][j] = eij;
        }
    }
    __syncthreads();

    filtering_accum_t accum[NUM_CHAN];
    for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
        for (int i = 0; i < NUM_CHAN; i++) {
            accum[i] = static_cast<filtering_accum_t>(0);
        }
        index_t first_ind = cuda::std::max(static_cast<index_t>(0), t - filter_phase_len + 1);
        index_t sample_idx = t * NUM_CHAN + NUM_CHAN - 1;
        index_t j_start = t;
        index_t h_ind { 0 };
        index_t niter = j_start - first_ind + 1;
        // For the last signal element, we need bounds-checking because we may need to zero-pad the signal.
        if (niter > 0) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                const filter_t h_val = (h_ind < filter_full_len) ? filter_acc(h_ind) : static_cast<filter_t>(0);
                if (sample_idx < input_len) {
                    detail::channelize_cmac(accum[chan],
                            detail::channelize_cast_filter<filtering_accum_t>(h_val),
                            detail::channelize_cast_input<filtering_accum_t>(input_b(sample_idx)));
                }
                h_ind++;
                sample_idx--;
            }
        }
        niter--;

        // The central elements require no bounds checking on the filter or signal.
        for (index_t i = 0; i < niter-1; i++) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                const filter_t h_val = filter_acc(h_ind);
                detail::channelize_cmac(accum[chan],
                        detail::channelize_cast_filter<filtering_accum_t>(h_val),
                        detail::channelize_cast_input<filtering_accum_t>(input_b(sample_idx)));
                h_ind++;
                sample_idx--;
            }
        }

        // For the first signal element / last filter tap, we need to bounds check the filter.
        if (niter > 0) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                if (h_ind >= filter_full_len) {
                    break;
                }
                const filter_t h_val = filter_acc(h_ind);
                detail::channelize_cmac(accum[chan],
                        detail::channelize_cast_filter<filtering_accum_t>(h_val),
                        detail::channelize_cast_input<filtering_accum_t>(input_b(sample_idx)));
                h_ind++;
                sample_idx--;
            }
        }


        // For complex inputs, the DFT will not generally be conjugate symmetric, so compute all
        // terms. For real inputs, we only compute the unique (up to conjugate symmetry) components.
        if constexpr (is_complex_v<input_t> || is_complex_half_v<input_t>) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                complex_accum_t dft { 0 };
                for (int j = 0; j < NUM_CHAN; j++) {
                    dft += accum[j] * smem_eij[chan][j];
                }
                output_b(t, static_cast<index_t>(chan)) = static_cast<output_t>(dft);
            }
        } else {
            constexpr int mid = NUM_CHAN/2 + 1;
            if constexpr (NUM_CHAN % 2 == 0) {
                // Channel 0, DC. There is no conjugate symmetric component for this value.
                {
                    complex_accum_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[0][j];
                    }
                    output_b(t, static_cast<index_t>(0)) = static_cast<output_t>(dft);
                }
                // Channel mid-1, Nyquist. There is no conjugate symmetric component for this value.
                {
                    complex_accum_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[mid-1][j];
                    }
                    output_b(t, static_cast<index_t>(mid-1)) = static_cast<output_t>(dft);
                }
                // Conjugate symmetric components
                for (int chan = 1; chan < mid-1; chan++) {
                    complex_accum_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[chan][j];
                    }
                    output_b(t, static_cast<index_t>(chan))             = static_cast<output_t>(dft);
                    output_b(t, static_cast<index_t>(NUM_CHAN - chan))  = static_cast<output_t>(conj(dft));
                }
            } else {
                // Channel 0, DC. There is no conjugate symmetric component for this value.
                {
                    complex_accum_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[0][j];
                    }
                    output_b(t, static_cast<index_t>(0)) = static_cast<output_t>(dft);
                }
                // Conjugate symmetric components
                for (int chan = 1; chan < mid; chan++) {
                    complex_accum_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[chan][j];
                    }
                    output_b(t, static_cast<index_t>(chan))             = static_cast<output_t>(dft);
                    output_b(t, static_cast<index_t>(NUM_CHAN - chan))  = static_cast<output_t>(conj(dft));
                }
            }
        }
    }
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

    const int tid = blockIdx.y * blockDim.x + threadIdx.x;

    const index_t num_elem_per_channel = inout.Size(ElemRank);
    const index_t num_channels = inout.Size(ChannelRank);

    const index_t mid = num_channels/2 + 1;
    if (tid >= num_elem_per_channel) {
        return;
    }

    // Bind batch coords; remaining dims are (elem, channel). Access with
    // inout_b(tid, chan).
    detail::TensorAccessor<DataType, IsUnitStride> inout_acc(inout);
    const auto batch_idx = BlockToIdx(inout, blockIdx.x, 2);
    auto inout_b = detail::bind_first_n<Rank - 2>(inout_acc, batch_idx);

    const index_t upper = (num_channels % 2 == 0) ? (mid - 1) : mid;
    for (index_t i = 1; i < upper; i++) {
        const value_t val = inout_b(static_cast<index_t>(tid), i);
        inout_b(static_cast<index_t>(tid), i) = conj(val);
        inout_b(static_cast<index_t>(tid), num_channels - i) = val;
    }
}

#endif // __CUDACC__

}; // namespace matx
