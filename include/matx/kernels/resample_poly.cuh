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

#include "cuComplex.h"
#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"

namespace matx {

#ifdef __CUDACC__ 
template <int THREADS, typename OutType, typename InType, typename FilterType>
__launch_bounds__(THREADS)
__global__ void ResamplePoly1D(OutType output, InType input, FilterType filter,
                    index_t up, index_t down)
{
    using output_t = typename OutType::scalar_type;
    using input_t = typename InType::scalar_type;
    using filter_t = typename FilterType::scalar_type;

    extern __shared__ __align__(alignof(double4)) uint8_t smem_filter[];
    filter_t *s_filter = reinterpret_cast<filter_t *>(smem_filter);

    constexpr int Rank = OutType::Rank();
    const index_t output_len = output.Size(Rank-1);
    index_t filter_len = filter.Size(0);
    const index_t input_len = input.Size(Rank-1);

    // We assume odd-length filters below. In the case of an even-length filter,
    // logically prepend the filter with a single zero to make its length odd.
    const bool is_even = filter_len % 2 == 0;
    if (filter_len % 2 == 0) {
        filter_len++;
    }

    const int phase_ind = blockIdx.y;
    const int tid = threadIdx.x;
    const index_t filter_len_half = filter_len/2;

    // All but the last dim are batch indices
    const int batch_idx = blockIdx.x;
    auto bdims = BlockToIdx(output, batch_idx, 1);

    // We assume odd-length filters in the below index calculations. If the filter
    // is even-length, then it has had a single zero logically prepended to it to
    // make it odd-length.
    // Consider resampling an input x by up=3, down=2. The upsampled and then
    // downsampled sequence -- ignoring filtering for now -- will look as folows.
    // | x0 |  0 |  0 | x1 |  0 |  0 | x2 | ... ]
    //
    //   d0        d1        d2        d3   ...
    // The dX are samples that will be kept after filtering and downsampling.
    // The filter is centered on the output sample. We want to avoid storing the
    // filter coefficents that will map to 0 in the upsampled sequence. Thus,
    // we see in the above that for filter h with length K and output d0, we will
    // need filter elements h[K/2], h[K/2 - 3], h[K/2 - 6], etc., depending upon
    // the length of the filter. The filter is flipped for convolution, which is
    // why the index is decreasing in the above. We call this set of coefficients
    // phase 0 because there is an offset of 0 between the xN points in the
    // upsampled sequence and the central tap of the filter.
    // Similarly, output d1 will use phase 1 coefficients because x1 is offset by
    // 1 from the central tap, and output d1 will use phase 2 coefficients because
    // x2 is offset by 2 from the central tap. Thus, there are up total phases and
    // therefore either filter_len / up or filter_len / up - 1 coefficients per phase.
    //
    // This kernel is launched with the phase index provided as a block index
    // and it will calculate the output points corresponding to this phase. We first
    // need to determine which filter samples we need for this phase and load them
    // into shared memory.
    // left_filter_ind is the filter index that corresponds to the input sample
    // in the upsampled sequence equal to or to the left of the output sample.
    // Thus, for phase 0, left_filter_ind is the central filter tap, h[k/2].
    // For phase 1, with our up=3, down=2 example, left_filter_ind is
    // h[K/2+2]. Once we have a single tap index and the corresponding x index,
    // we will stride in both sequences (input and filter) by +/- up to cover
    // the full filter phase.
    const index_t left_filter_ind = filter_len_half + (phase_ind*down) % up;

    // If left_filter_ind >= filter_len, then the filter is not long enough to reach
    // a potentially non-zero sample value on the left. In that case, set the
    // last_filter_ind to the next non-zero sample to the right of the output
    // index.    
    const index_t last_filter_ind = (left_filter_ind < filter_len) ?
        left_filter_ind + up * ((filter_len - 1 - left_filter_ind) / up) :
        left_filter_ind - up;
        
    // If last_filter_ind is now < 0, that means that the filter does not have
    // any corresponding non-zero samples (i.e. samples from the input signal).
    // Thus, all output values for this phase are zero because the filter only
    // overlaps with zero-filled samples from the upsampling operation.
    if (last_filter_ind < 0) {
        for (index_t out_ind = phase_ind + tid * up; out_ind < output_len; out_ind += THREADS * up) {
            bdims[Rank - 1] = out_ind;
            output.operator()(bdims) = 0;
        }
        return;
    }

    const index_t first_filter_ind = left_filter_ind - up * (left_filter_ind / up);
    const index_t this_phase_len = (last_filter_ind - first_filter_ind)/up + 1;

    // Scale the filter coefficients by up to match scipy's convention
    const filter_t scale = static_cast<filter_t>(up);

    // Flip the filter when writing to smem. The filtering is a convolution.
    if (is_even) {
        for (index_t t = tid; t < this_phase_len; t += THREADS) {
            const index_t ind = t * up + first_filter_ind;
            const index_t smem_ind = this_phase_len - 1 - t;
            s_filter[smem_ind] = (ind > 0) ? scale * filter.operator()(ind-1) : static_cast<filter_t>(0);
        }
    } else {
        for (index_t t = tid; t < this_phase_len; t += THREADS) {
            const index_t ind = t * up + first_filter_ind;
            const index_t smem_ind = this_phase_len - 1 - t;
            s_filter[smem_ind] = scale * filter.operator()(ind);
        }
    }


    __syncthreads();

    // left_h_ind is the index in s_filter that contains the filter tap that will be applied to the
    // last input signal value not to the right of the output index in the virtual upsampled array.
    // If the filter has odd length and a given output value aligns with an input value, then
    // left_h_ind would reference the central tap. If the output value corresponds to a zero-padded
    // value (i.e. a 0 inserted during upsampling), then left_h_ind is the filter tap applied
    // to the nearest input value to the left of this output value.
    const index_t left_h_ind = (last_filter_ind - left_filter_ind)/up;

    const index_t max_h_epilogue = this_phase_len - left_h_ind - 1;
    const index_t max_input_ind = static_cast<int>(input_len) - 1;

    for (index_t out_ind = phase_ind + tid * up; out_ind < output_len; out_ind += THREADS * up) {
        // out_ind is the index in the output array and up_ind is the corresponding
        // index in the upsampled array
        const index_t up_ind = out_ind * down;

        // input_ind is the largest index in the input array that is not greater than
        // (to the right of, in the previous figure earlier) up_ind.
        const index_t input_ind = up_ind / up;

        // We want x_ind and h_ind to be the first aligned input and filter samples
        // of the convolution and n to be the number of taps. prologue is the number
        // of valid samples before input_ind. In the case that the filter is not
        // long enough to include input_ind, last_filter_ind is left_filter_ind - up
        // and thus left_h_ind and prologue are both -1.
        const index_t prologue = std::min(input_ind, left_h_ind);
        // epilogue is the number of valid samples after input_ind.
        const index_t epilogue = std::min(max_input_ind - input_ind, max_h_epilogue);
        // n is the number of valid samples. If input_ind is not valid because it
        // precedes the reach of the filter, then prologue = -1 and n is just the
        // epilogue.
        const index_t n = prologue + 1 + epilogue;

        // Finally, convolve the filter and input samples
        index_t x_ind = input_ind - prologue;
        index_t h_ind = left_h_ind - prologue;
        output_t accum {};
        for (index_t j = 0; j < n; j++) {
            bdims[Rank - 1] = x_ind++;
            accum += s_filter[h_ind++] * input.operator()(bdims);
        }

        bdims[Rank - 1] = out_ind;
        output.operator()(bdims) = accum;
    }
}

#endif // __CUDACC__

}; // namespace matx