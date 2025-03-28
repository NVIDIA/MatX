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
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"

namespace matx {

// Number of output elements generated per thread
constexpr index_t CHANNELIZE_POLY1D_ELEMS_PER_THREAD = 1;

#ifdef __CUDACC__ 
template <int THREADS, typename OutType, typename InType, typename FilterType>
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D(OutType output, InType input, FilterType filter)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;

    // Opportunistically store the filter taps in shared memory if the static shared memory
    // size is sufficient. Otherwise, we will read directly from global memory on use.
    const int SMEM_MAX_FILTER_TAPS = 128;
    // Versions of CUDA prior to 11.8 do not allow static shared memory allocations of
    // cuda::std::complex types due to it having no trivial constructor. This workaround
    // prevents an 'initializer not allowed for __shared__ variable' error.
    __align__(sizeof(filter_t)) __shared__ uint8_t smem_filter_workaround[sizeof(filter_t)*SMEM_MAX_FILTER_TAPS];
    filter_t (&smem_filter)[SMEM_MAX_FILTER_TAPS] = reinterpret_cast<filter_t (&)[SMEM_MAX_FILTER_TAPS]>(smem_filter_workaround);

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

    constexpr index_t ELEMS_PER_BLOCK = CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
    const index_t first_out_elem = elem_block * CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
    const index_t last_out_elem = cuda::std::min(
        output_len_per_channel - 1, first_out_elem + ELEMS_PER_BLOCK - 1);

    if (filter_phase_len <= SMEM_MAX_FILTER_TAPS) {
        for (index_t t = tid; t < filter_phase_len-1; t += THREADS) {
            const index_t h_ind = channel + t * num_channels;
            smem_filter[t] = filter.operator()(h_ind);
        }
        if (tid == THREADS-1) {
            const index_t h_ind = channel + (filter_phase_len-1) * num_channels;
            smem_filter[filter_phase_len-1] = (h_ind < filter_full_len) ?
                filter.operator()(h_ind) : static_cast<filter_t>(0);
        }

        __syncthreads();
    }

    auto indims = BlockToIdx(input, blockIdx.z, 1);
    auto outdims = BlockToIdx(output, blockIdx.z, 2);
    outdims[ChannelRank] = channel;

    if (filter_phase_len <= SMEM_MAX_FILTER_TAPS) {
        for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
            const index_t first_ind = cuda::std::max(static_cast<index_t>(0), t - filter_phase_len + 1);
            output_t accum {};
            const filter_t *h = smem_filter;
            // index_t in MatX should be signed (32 or 64 bit), so j-- below will not underflow
            static_assert(std::is_signed_v<index_t>, "assumed signed index_t, but it is unsigned");
            indims[InRank-1] = t * num_channels + (num_channels - 1 - channel);
            index_t j_start = t;
            if (indims[InRank-1] >= input_len) {
                j_start--;
                indims[InRank-1] -= num_channels;
                h++;
            }
            // Because the filter must fit in shared memory, we know that the maximum
            // number of iterations fits well within an int.
            const int niter = static_cast<int>(j_start - first_ind + 1);
            input_t in_val;
            for (int i = 0; i < niter; i++) {
                cuda::std::apply([&in_val, &input](auto &&...args) {                    
                    in_val = input.operator()(args...);
                }, indims);
                accum += (*h) * in_val;
                indims[InRank-1] -= num_channels;
                h++;
            }
            outdims[OutElemRank] = t;
            cuda::std::apply([accum, &output](auto &&...args) {
                output.operator()(args...) = accum;
            }, outdims);
        }
    } else {
        for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
            index_t first_ind = cuda::std::max(static_cast<index_t>(0), t - filter_phase_len + 1);
            // If we use the last filter tap for this phase (which is the first index because
            // the filter is flipped), then it may be a padded zero. If so, increment first_ind
            // by 1 to avoid using the zero. This prevents a bounds-check in the inner loop.
            if (first_ind == (t - filter_phase_len + 1)) {
                const bool h_is_padded = ((filter_phase_len-1) * num_channels + channel) >= filter_full_len;
                if (h_is_padded) {
                    first_ind++;
                }
            }
            indims[InRank-1] = t * num_channels + (num_channels - 1 - channel);
            index_t j_start = t;
            index_t h_ind { channel };
            // If the last signal element is a zero-pad value, then skip it to prevent needing
            // per-access bounds checking in the inner loop.
            if (indims[InRank-1] >= input_len) {
                j_start--;
                indims[InRank-1] -= num_channels;
                h_ind += num_channels;
            }
            const index_t niter = j_start - first_ind + 1;
            output_t accum {};
            input_t in_val;
            for (index_t i = 0; i < niter; i++) {
                cuda::std::apply([&in_val, &input](auto &&...args) {                    
                    in_val = input.operator()(args...);
                }, indims);
                const filter_t h_val = filter.operator()(h_ind);
                accum += h_val * in_val;
                h_ind += num_channels;
                indims[InRank-1] -= num_channels;
            }
            outdims[OutElemRank] = t;
            cuda::std::apply([accum, &output](auto &&...args) {
                output.operator()(args...) = accum;
            }, outdims);
        }
    }
}

// This kernel works in cases where the full filter (with potentially some zero padding) and
// the inputs required to compute elems_per_channel_per_cta outputs all fit into shared memory.
template <typename OutType, typename InType, typename FilterType>
__global__ void ChannelizePoly1D_Smem(OutType output, InType input, FilterType filter, index_t elems_per_channel_per_cta)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;

    extern __shared__ uint8_t __attribute((aligned(16))) smem_dyn_align16[];

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
    const uint32_t num_channels = static_cast<uint32_t>(output.Size(ChannelRank));
    const uint32_t filter_full_len = static_cast<uint32_t>(filter.Size(0));
    const uint32_t filter_phase_len = static_cast<uint32_t>((filter_full_len + num_channels - 1) / num_channels);

    filter_t *smem_h = reinterpret_cast<filter_t *>(smem_dyn_align16);
    size_t smem_input_offset = sizeof(filter_t) * filter_phase_len * num_channels;
    if (smem_input_offset % sizeof(input_t)) {
        smem_input_offset += sizeof(input_t) - smem_input_offset % sizeof(input_t);
    }
    input_t *smem_input = reinterpret_cast<input_t *>(smem_dyn_align16 + smem_input_offset);

    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t nthreads = blockDim.x * blockDim.y;
    const uint32_t chan = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t by = blockDim.y;

    for (uint32_t t = tid; t < filter_full_len; t += nthreads) {
        smem_h[t] = filter.operator()(t);
    }

    for (uint32_t t = filter_full_len+tid; t < filter_phase_len * num_channels; t += nthreads) {
        smem_h[t] = static_cast<filter_t>(0);
    }

    // The input stored in shared memory is logically [smem_input_height, num_channels] where
    // smem_input_height is the number of samples at the output sample rate stored in smem.
    const uint32_t smem_input_height = filter_phase_len + by - 1;

    const index_t start_elem = blockIdx.x * elems_per_channel_per_cta;
    const index_t last_elem_this_block = static_cast<index_t>(blockIdx.x) * elems_per_channel_per_cta + (elems_per_channel_per_cta - 1);
    const index_t last_elem = cuda::std::min(output_len_per_channel-1, last_elem_this_block);
    auto indims = BlockToIdx(input, blockIdx.z, 1);
    auto outdims = BlockToIdx(output, blockIdx.z, 2);
    outdims[ChannelRank] = chan;

    for (uint32_t t = ty; t < filter_phase_len-1; t += by) {
        const index_t out_sample_ind = start_elem - (filter_phase_len-1) + t;
        const uint32_t smem_ind = t * num_channels + chan;
        const index_t input_ind = out_sample_ind * num_channels + chan;
        if (input_ind >= 0 && input_ind < input_len) {
            indims[InRank-1] = input_ind;
            cuda::std::apply([smem_input, smem_ind, &input](auto &&...args) {
                smem_input[smem_ind] = input.operator()(args...);
            }, indims);
        } else {
            smem_input[smem_ind] = static_cast<filter_t>(0);
        }
    }

    index_t next_start_elem = start_elem;
    const index_t num_elem_iters = (last_elem - start_elem + 1 + by - 1) / by;

    uint32_t cached_input_ind_tail = filter_phase_len - 1 + ty;
    const filter_t *h_start = smem_h + num_channels * filter_phase_len - (num_channels - chan);
    for (index_t iter = 0; iter < num_elem_iters; iter++) {

        __syncthreads();

        // Load next elems_per_channel_per_cta elements for each channel
        const index_t next_last_elem = cuda::std::min(next_start_elem + static_cast<index_t>(by) - 1, last_elem);
        const uint32_t out_samples_this_iter = static_cast<uint32_t>(next_last_elem - next_start_elem + 1);
        if (ty < out_samples_this_iter) {
            indims[InRank-1] = (next_start_elem + ty) * num_channels + chan;
            const uint32_t smem_ind = cached_input_ind_tail * num_channels + chan;
            if (indims[InRank-1] < input_len) {
                cuda::std::apply([smem_input, smem_ind, &input](auto &&...args) {
                    smem_input[smem_ind] = input.operator()(args...);
                }, indims);
            } else {
                smem_input[smem_ind] = static_cast<filter_t>(0);
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

        outdims[OutElemRank] = next_start_elem + ty;
        if (outdims[OutElemRank] <= last_elem) {
            const filter_t *h = h_start;
            output_t accum { 0 };
            const uint32_t first_end = cuda::std::min(cached_input_ind_tail + filter_phase_len - 1, smem_input_height - 1);
            // The footprint of samples involved in the convolution may wrap from the end
            // to the beginning of smem_input. The prologue below handles the samples from
            // the current tail to the end of smem_input and the epilogue starts back at the
            // beginning of smem_input.
            const uint32_t prologue_count = (first_end - cached_input_ind_tail + 1);
            const uint32_t epilogue_count = (prologue_count < filter_phase_len) ? filter_phase_len - prologue_count : 0;
            const input_t *sample = smem_input + cached_input_ind_tail * num_channels + (num_channels - 1 - chan);
            // Apply the filter h in reverse order below to flip the filter for convolution
            for (uint32_t k = 0; k < prologue_count; k++) {
                accum += (*h) * (*sample);
                sample += num_channels;
                h -= num_channels;
            }
            sample = smem_input + (num_channels - 1 - chan);
            for (uint32_t k = 0; k < epilogue_count; k++) {
                accum += (*h) * (*sample);
                sample += num_channels;
                h -= num_channels;
            }

            cuda::std::apply([accum, &output](auto &&...args) {
                output.operator()(args...) = accum;
            }, outdims);
        }

        next_start_elem += out_samples_this_iter;
    }
}

template <int THREADS, int NUM_CHAN, typename OutType, typename InType, typename FilterType>
__launch_bounds__(THREADS)
__global__ void ChannelizePoly1D_FusedChan(OutType output, InType input, FilterType filter)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;

    constexpr int InRank = InType::Rank();
    constexpr int OutRank = OutType::Rank();
    constexpr int ChannelRank = OutRank-1;
    constexpr int OutElemRank = OutRank-2;

    const index_t input_len = input.Size(InRank-1);
    const index_t output_len_per_channel = output.Size(OutElemRank);
    const index_t filter_full_len = filter.Size(0);
    const index_t filter_phase_len = (filter_full_len + NUM_CHAN - 1) / NUM_CHAN;

    const int elem_block = blockIdx.x;
    const int tid = threadIdx.x;

    auto indims = BlockToIdx(input, blockIdx.z, 1);
    auto outdims = BlockToIdx(output, blockIdx.z, 2);

    constexpr index_t ELEMS_PER_BLOCK = CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
    const index_t first_out_elem = elem_block * CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
    const index_t last_out_elem = cuda::std::min(
        output_len_per_channel - 1, first_out_elem + ELEMS_PER_BLOCK - 1);

    // Versions of CUDA prior to 11.8 do not allow static shared memory allocations of
    // cuda::std::complex types due to it having no trivial constructor. This workaround
    // prevents an 'initializer not allowed for __shared__ variable' error.
    __align__(sizeof(output_t)) __shared__ uint8_t smem_eij_workaround[sizeof(output_t)*NUM_CHAN*NUM_CHAN];
    output_t (&smem_eij)[NUM_CHAN][NUM_CHAN] = reinterpret_cast<output_t (&)[NUM_CHAN][NUM_CHAN]>(smem_eij_workaround);
    // Pre-compute the DFT complex exponentials and store in shared memory
    for (int t = tid; t < NUM_CHAN*NUM_CHAN; t += THREADS) {
        const int i = t / NUM_CHAN;
        const int j = t % NUM_CHAN;
        if constexpr (std::is_same_v<output_t, std::complex<double>> || std::is_same_v<output_t, cuda::std::complex<double>>) {
            const double arg = 2.0 * M_PI * j * i / NUM_CHAN;
            double sinx, cosx;
            sincos(arg, &sinx, &cosx);
            output_t eij { cosx, sinx };
            smem_eij[i][j] = eij;
        } else {
            const float arg = 2.0f * static_cast<float>(M_PI) * j * i / NUM_CHAN;
            float sinx, cosx;
            sincosf(arg, &sinx, &cosx);
            output_t eij { cosx, sinx };
            smem_eij[i][j] = eij;
        }
    }
    __syncthreads();

    output_t accum[NUM_CHAN];
    for (index_t t = first_out_elem+tid; t <= last_out_elem; t += THREADS) {
        for (int i = 0; i < NUM_CHAN; i++) {
            accum[i] = static_cast<output_t>(0);
        }
        index_t first_ind = cuda::std::max(static_cast<index_t>(0), t - filter_phase_len + 1);
        indims[InRank-1] = t * NUM_CHAN + NUM_CHAN - 1;
        index_t j_start = t;
        index_t h_ind { 0 };
        index_t niter = j_start - first_ind + 1;
        // For the last signal element, we need bounds-checking because we may need to zero-pad the signal.
        if (niter > 0) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                const filter_t h_val = (h_ind < filter_full_len) ? filter.operator()(h_ind) : static_cast<filter_t>(0);
                if (indims[InRank-1] < input_len) {
                    cuda::std::apply([&accum, chan, h_val, &input](auto &&...args) {                    
                        accum[chan] += h_val * input.operator()(args...);
                    }, indims);
                }
                h_ind++;
                indims[InRank-1]--;
            }        
        }
        niter--;

        // The central elements require no bounds checking on the filter or signal.
        for (index_t i = 0; i < niter-1; i++) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                const filter_t h_val = filter.operator()(h_ind);
                cuda::std::apply([&accum, chan, h_val, &input](auto &&...args) {                    
                    accum[chan] += h_val * input.operator()(args...);
                }, indims);
                h_ind++;
                indims[InRank-1]--;
            }
        }

        // For the first signal element / last filter tap, we need to bounds check the filter.
        if (niter > 0) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                if (h_ind >= filter_full_len) {
                    break;
                }
                // const filter_t h_val = (h_ind < filter_full_len) ? filter.operator()(h_ind) : static_cast<filter_t>(0);
                const filter_t h_val = filter.operator()(h_ind);
                cuda::std::apply([&accum, chan, h_val, &input](auto &&...args) {                    
                    accum[chan] += h_val * input.operator()(args...);
                }, indims);
                h_ind++;
                indims[InRank-1]--;
            }
        }

        outdims[OutElemRank] = t;

        // For complex inputs, the DFT will not generally be conjugate symmetric, so compute all
        // terms. For real inputs, we only compute the unique (up to conjugate symmetry) components.
        if constexpr (is_complex_v<input_t> || is_complex_half_v<input_t>) {
            for (int chan = 0; chan < NUM_CHAN; chan++) {
                output_t dft { 0 };
                for (int j = 0; j < NUM_CHAN; j++) {
                    dft += accum[j] * smem_eij[chan][j];
                }
                outdims[ChannelRank] = chan;
                cuda::std::apply([dft, &output](auto &&...args) {
                    output.operator()(args...) = dft;
                }, outdims);            
            }
        } else {
            constexpr int mid = NUM_CHAN/2 + 1;
            if constexpr (NUM_CHAN % 2 == 0) {
                // Channel 0, DC. There is no conjugate symmetric component for this value.
                {
                    output_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[0][j];
                    }
                    outdims[ChannelRank] = 0;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = dft;
                    }, outdims);
                }
                // Channel mid-1, Nyquist. There is no conjugate symmetric component for this value.
                {
                    output_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[mid-1][j];
                    }
                    outdims[ChannelRank] = mid-1;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = dft;
                    }, outdims);
                }
                // Conjugate symmetric components
                for (int chan = 1; chan < mid-1; chan++) {
                    output_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[chan][j];
                    }
                    outdims[ChannelRank] = chan;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = dft;
                    }, outdims);
                    outdims[ChannelRank] = NUM_CHAN - chan;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = conj(dft);
                    }, outdims);
                }
            } else {
                // Channel 0, DC. There is no conjugate symmetric component for this value.
                {
                    output_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[0][j];
                    }
                    outdims[ChannelRank] = 0;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = dft;
                    }, outdims);
                }
                // Conjugate symmetric components
                for (int chan = 1; chan < mid; chan++) {
                    output_t dft { 0 };
                    for (int j = 0; j < NUM_CHAN; j++) {
                        dft += accum[j] * smem_eij[chan][j];
                    }
                    outdims[ChannelRank] = chan;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = dft;
                    }, outdims);
                    outdims[ChannelRank] = NUM_CHAN - chan;
                    cuda::std::apply([dft, &output](auto &&...args) {
                        output.operator()(args...) = conj(dft);
                    }, outdims);
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
template <typename DataType>
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

    auto dims = BlockToIdx(inout, blockIdx.x, 2);
    dims[ElemRank] = tid;
    value_t val;
    if (num_channels % 2 == 0) {
        for (index_t i = 1; i < mid-1; i++) {
            dims[ChannelRank] = i;
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                val = inout.operator()(args...);
            }, dims);
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                inout.operator()(args...) = conj(val);
            }, dims);
            dims[ChannelRank] = num_channels - i;
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                inout.operator()(args...) = val;
            }, dims);            
        }
    } else {
        for (index_t i = 1; i < mid; i++) {
            dims[ChannelRank] = i;
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                val = inout.operator()(args...);
            }, dims);
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                inout.operator()(args...) = conj(val);
            }, dims);
            dims[ChannelRank] = num_channels - i;
            cuda::std::apply([&val, &inout](auto &&...args) {                    
                inout.operator()(args...) = val;
            }, dims);            
        }
    }
}

#endif // __CUDACC__

}; // namespace matx