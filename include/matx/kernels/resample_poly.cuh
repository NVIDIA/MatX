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

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
#endif

#include "cuComplex.h"
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/max.h>
#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/kernels/tensor_accessor.h"

namespace matx {

// Use for __launch_bounds__ to allow the compiler to tune register usage
static constexpr int MATX_RESAMPLE_POLY_MAX_NUM_THREADS = 256;

// We use a static 11 KiB buffer to potentially store the filter. If it fits, we will load it
// into smem. If not, then we will load it from global memory at the time of use. We choose
// 11 KiB so that we can definitely fit four blocks in 48 KiB, leaving 1 KiB per block
// for the driver.
static constexpr size_t MATX_RESAMPLE_POLY_MAX_SMEM_BYTES = 11*1024;

#ifdef __CUDACC__ 

template <int THREADS, typename FilterAcc>
__device__ inline void ResamplePoly1D_LoadFilter(typename FilterAcc::value_type *s_filter,
                                                  const FilterAcc &filter_acc, index_t filter_len)
{
    // FilterAcc is a TensorAccessor / BoundAccessor wrapping the filter
    // operator; both expose `value_type` aliased from the underlying op's
    // value_type. The static_assert documents that contract so a refactor
    // that drops `value_type` from those wrappers fails here at the helper
    // site rather than silently breaking smem-filter loads.
    using filter_value_t = typename FilterAcc::value_type;
    static_assert(cuda::std::is_arithmetic_v<filter_value_t> ||
                  is_complex_v<filter_value_t> ||
                  is_matx_type_v<filter_value_t>,
                  "ResamplePoly1D_LoadFilter requires FilterAcc::value_type "
                  "to be a MatX-supported numeric scalar/complex type");
    const int tid = threadIdx.x;
    if (filter_len % 2 == 0) {
        for (int t = tid; t < filter_len; t += THREADS) {
            s_filter[t+1] = filter_acc(t);
        }
        if (tid == 0) {
            s_filter[0] = static_cast<filter_value_t>(0);
        }
    } else {
        for (int t = tid; t < filter_len; t += THREADS) {
            s_filter[t] = filter_acc(t);
        }
    }
    __syncthreads();    
}

// out_offset shifts the *global* output index used for the polyphase math while
// the write position stays local, so this kernel can emit an arbitrary window
// [out_offset, out_offset + output.Size(Rank-1)) of the full resample grid
// (used by the streaming object to compute only the outputs it owns). It is 0
// for a normal full-grid call, in which case global index == write position.
template <int THREADS, bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename index_t>
__launch_bounds__(MATX_RESAMPLE_POLY_MAX_NUM_THREADS)
__global__ void ResamplePoly1D_ElemBlock(OutType output, InType input, FilterType filter,
                    index_t up, index_t down, index_t elems_per_thread, index_t out_offset)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;

    extern __shared__ uint8_t smem_filter[];
    filter_t *s_filter = reinterpret_cast<filter_t *>(smem_filter);

    constexpr int Rank = OutType::Rank();
    const index_t output_len = static_cast<index_t>(output.Size(Rank-1));
    index_t filter_len = static_cast<index_t>(filter.Size(0));
    const index_t input_len = static_cast<index_t>(input.Size(Rank-1));

    const size_t filter_sz_bytes = (filter_len % 2 == 0) ? sizeof(filter_t)*(filter_len+1) : sizeof(filter_t)*filter_len;
    const bool load_filter_to_smem = (filter_sz_bytes <= MATX_RESAMPLE_POLY_MAX_SMEM_BYTES);

    const int elem_block = blockIdx.z;
    const int tid = threadIdx.x;
    // const int THREADS = blockDim.x;

    // Wrap input/output/filter in TensorAccessors, then bind the leading Rank-1
    // batch dims once (below). On the unit-stride fast path per-access collapses
    // to base_ptr[idx] arithmetic; otherwise it forwards to operator() and works
    // for any MatX op. Output and input share the same batch shape (asserted by
    // resample_poly_impl), so a single BlockToIdx call feeds both bindings.
    detail::TensorAccessor<InType,     IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType,    IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);

    if (load_filter_to_smem) {
        ResamplePoly1D_LoadFilter<THREADS>(s_filter, filter_acc, filter_len);
        if (filter_len % 2 == 0) {
            filter_len++;
        }
    }

    // Bind the leading Rank-1 batch dims into both accessors. Input and
    // output share their batch shape, so a single BlockToIdx call feeds both.
    const int batch_idx = blockIdx.x;
    const auto batch_idx_arr = BlockToIdx(output, batch_idx, 1);
    auto input_b  = detail::bind_first_n<Rank - 1>(input_acc,  batch_idx_arr);
    auto output_b = detail::bind_first_n<Rank - 1>(output_acc, batch_idx_arr);

    // Scale the filter coefficients by up to match scipy's convention
    const filter_t scale = static_cast<filter_t>(up);
    const index_t max_input_ind = input_len - 1;

    const index_t filter_len_half = filter_len/2;
    // The loops below assume odd-length filters with a central tap. In the case of storing an
    // even-length filter to smem, a zero is pre-pended to the filter (prior to flipping for convolution)
    // so that the stored filter length is always odd-length.
    // Thus, for a stored filter, both filter_len/2 and (filter_len-1)/2 reference the central tap.
    // In the case of an originally even-length filter, the index of the central tap in the filter
    // tensor is filter_len/2 - 1. When not storing the filter to smem, we want the same central
    // tap, so we compute the index as (filter_len-1)/2. This will return the same result for
    // natively odd-length filters, but for even-length filters will reference the same coefficient
    // whether or not the filter has been loaded to shared memory.
    const index_t filter_central_tap = (filter_len-1)/2;
    const index_t start_ind = elem_block * elems_per_thread * THREADS + tid;
    const index_t last_ind = cuda::std::min(output_len - 1, start_ind + (elems_per_thread-1) * THREADS);
    if (load_filter_to_smem) {
        for (index_t out_ind = start_ind; out_ind <= last_ind; out_ind += THREADS) {
            const index_t up_ind = (out_ind + out_offset) * down;
            const index_t up_start = cuda::std::max(static_cast<index_t>(0), up_ind - filter_len_half);
            const index_t up_end = cuda::std::min(max_input_ind * up, up_ind + filter_len_half);
            const index_t x_start = (up_start + up - 1) / up;
            index_t x_end = up_end / up;
            // Since the filter is in shared memory, we can narrow the index type to 32 bits
            int h_ind = static_cast<int>(filter_central_tap + (up_ind - up*x_start));

            output_t accum {};
            // Cap the inner-loop unroll at 8; unroll=16 can saturate the MIO
            // pipeline and reduce performance.
            #pragma unroll 8
            for (index_t i = x_start; i <= x_end; i++) {
                const input_t in_val = input_b(i);
                accum += in_val * s_filter[h_ind];
                h_ind -= up;
            }

            accum *= scale;
            output_b(out_ind) = accum;
        }
    } else {
        for (index_t out_ind = start_ind; out_ind <= last_ind; out_ind += THREADS) {
            const index_t up_ind = (out_ind + out_offset) * down;
            const index_t up_start = cuda::std::max(static_cast<index_t>(0), up_ind - filter_len_half);
            const index_t up_end = cuda::std::min(max_input_ind * up, up_ind + filter_len_half);
            const index_t x_start = (up_start + up - 1) / up;
            index_t x_end = up_end / up;
            index_t h_ind = filter_central_tap + (up_ind - up*x_start);
            if (h_ind - up*(x_end-x_start) < 0) {
                x_end--;
            }

            output_t accum {};
            for (index_t i = x_start; i <= x_end; i++) {
                const input_t in_val = input_b(i);
                accum += in_val * filter_acc(h_ind);
                h_ind -= up;
            }

            accum *= scale;
            output_b(out_ind) = accum;
        }
    }

}

// See ResamplePoly1D_ElemBlock for the out_offset window semantics.
template <int THREADS, bool IsUnitStride, typename OutType, typename InType, typename FilterType, typename index_t>
__launch_bounds__(MATX_RESAMPLE_POLY_MAX_NUM_THREADS)
__global__ void ResamplePoly1D_WarpCentric(OutType output, InType input, FilterType filter,
                    index_t up, index_t down, index_t elems_per_warp, index_t out_offset)
{
    using output_t = typename OutType::value_type;
    using input_t = typename InType::value_type;
    using filter_t = typename FilterType::value_type;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<WARP_SIZE>(block);
    const int warp_id = tile.meta_group_rank();
    const int NUM_WARPS = THREADS / WARP_SIZE;
    const int lane_id = tile.thread_rank();

    extern __shared__ uint8_t smem_filter[];
    filter_t *s_filter = reinterpret_cast<filter_t *>(smem_filter);

    constexpr int Rank = OutType::Rank();
    const index_t output_len = static_cast<index_t>(output.Size(Rank-1));
    index_t filter_len = static_cast<index_t>(filter.Size(0));
    const index_t input_len = static_cast<index_t>(input.Size(Rank-1));    

    const size_t filter_sz_bytes = (filter_len % 2 == 0) ? sizeof(filter_t)*(filter_len+1) : sizeof(filter_t)*filter_len;
    const bool load_filter_to_smem = (filter_sz_bytes <= MATX_RESAMPLE_POLY_MAX_SMEM_BYTES);

    const int elem_block = blockIdx.z;

    // TensorAccessors with per-block batch binding (see ResamplePoly1D_ElemBlock).
    detail::TensorAccessor<InType,     IsUnitStride> input_acc(input);
    detail::TensorAccessor<OutType,    IsUnitStride> output_acc(output);
    detail::TensorAccessor<FilterType, IsUnitStride> filter_acc(filter);

    if (load_filter_to_smem) {
        ResamplePoly1D_LoadFilter<THREADS>(s_filter, filter_acc, filter_len);
        if (filter_len % 2 == 0) {
            filter_len++;
        }        
    }

    // Bind the leading Rank-1 batch dims into both accessors. Input and
    // output share their batch shape, so a single BlockToIdx call feeds both.
    const int batch_idx = blockIdx.x;
    const auto batch_idx_arr = BlockToIdx(output, batch_idx, 1);
    auto input_b  = detail::bind_first_n<Rank - 1>(input_acc,  batch_idx_arr);
    auto output_b = detail::bind_first_n<Rank - 1>(output_acc, batch_idx_arr);

    // Scale the filter coefficients by up to match scipy's convention
    const filter_t scale = static_cast<filter_t>(up);
    const index_t max_input_ind = input_len - 1;

    const index_t filter_len_half = filter_len/2;
    const index_t filter_central_tap = (filter_len-1)/2;
    const index_t start_ind = elem_block * elems_per_warp * NUM_WARPS;
    const index_t last_ind = cuda::std::min(output_len - 1, start_ind + elems_per_warp * NUM_WARPS - 1);
    if (load_filter_to_smem) {
        for (index_t out_ind = start_ind+warp_id; out_ind <= last_ind; out_ind += NUM_WARPS) {
            const index_t up_ind = (out_ind + out_offset) * down;
            const index_t up_start = cuda::std::max(static_cast<index_t>(0), up_ind - filter_len_half);
            const index_t up_end = cuda::std::min(max_input_ind * up, up_ind + filter_len_half);
            const index_t x_start = (up_start + up - 1) / up;
            index_t x_end = up_end / up;
            // Since the filter is in shared memory, we can narrow the index type to 32 bits
            int h_ind = static_cast<int>(filter_central_tap + (up_ind - up*x_start)) - static_cast<int>(lane_id*up);

            output_t accum {};
            for (index_t i = x_start+lane_id; i <= x_end; i += WARP_SIZE) {
                const input_t in_val = input_b(i);
                accum += in_val * s_filter[h_ind];
                h_ind -= up * WARP_SIZE;
            }

            accum *= scale;
            if constexpr (is_complex_v<output_t>) {
                using inner_type = typename inner_op_type_t<output_t>::type;
                accum.real(cg::reduce(tile, accum.real(), cg::plus<inner_type>()));
                accum.imag(cg::reduce(tile, accum.imag(), cg::plus<inner_type>()));
            } else {
                accum = cg::reduce(tile, accum, cg::plus<output_t>());
            }
            if (lane_id == 0) {
                output_b(out_ind) = accum;
            }
        }
    } else {
        for (index_t out_ind = start_ind+warp_id; out_ind <= last_ind; out_ind += NUM_WARPS) {
            const index_t up_ind = (out_ind + out_offset) * down;
            const index_t up_start = cuda::std::max(static_cast<index_t>(0), up_ind - filter_len_half);
            const index_t up_end = cuda::std::min(max_input_ind * up, up_ind + filter_len_half);
            const index_t x_start = (up_start + up - 1) / up;
            index_t x_end = up_end / up;
            index_t h_ind = filter_central_tap + (up_ind - up*x_start);
            if (h_ind - up*(x_end-x_start) < 0) {
                x_end--;
            }
            h_ind -= lane_id*up;

            output_t accum {};
            for (index_t i = x_start+lane_id; i <= x_end; i += WARP_SIZE) {
                const input_t in_val = input_b(i);
                accum += in_val * filter_acc(h_ind);
                h_ind -= up * WARP_SIZE;
            }

            accum *= scale;
            if constexpr (is_complex_v<output_t>) {
                using inner_type = typename inner_op_type_t<output_t>::type;
                accum.real(cg::reduce(tile, accum.real(), cg::plus<inner_type>()));
                accum.imag(cg::reduce(tile, accum.imag(), cg::plus<inner_type>()));
            } else {
                accum = cg::reduce(tile, accum, cg::plus<output_t>());
            }
            if (lane_id == 0) {
                output_b(out_ind) = accum;
            }
        }
    }
}

#endif // __CUDACC__

}; // namespace matx
