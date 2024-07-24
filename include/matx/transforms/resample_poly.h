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

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/operators/clone.h"
#include "matx/kernels/resample_poly.cuh"

namespace matx {
namespace detail {

template <typename OutType, typename InType, typename FilterType>
inline void matxResamplePoly1DInternal(OutType &o, const InType &i,
                                     const FilterType &filter, index_t up, index_t down,
                                     cudaStream_t stream)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  using output_t = typename OutType::value_type;
  using shape_type = typename OutType::shape_type;

  // Even-length filters will be prepended with a single 0 to make them odd-length
  const shape_type filter_len = filter.Size(FilterType::Rank()-1);
  const index_t max_phase_len = (filter_len % 2 == 0) ?
    ((filter_len + 1 + up - 1) / up) :
    ((filter_len + up - 1) / up);

  auto downcast_to_32b_index = [&i, filter_len, up, down]() -> bool {
      if constexpr (sizeof(index_t) == 4) {
        // The index is already 32 bits
        return false;
      } else {
        return
          // + 1 because we may include a zero padded after the last input element
          (i.Size(i.Rank() - 1)+1) * up <= std::numeric_limits<int32_t>::max() &&
          (filter_len+1) <= std::numeric_limits<int32_t>::max() &&
          down <= std::numeric_limits<int32_t>::max();
      }
  };

  const index_t output_len = o.Size(OutType::Rank()-1);

  // We default to the ElemBlock kernel as it tends to work well for general problems.
  enum class ResampleKernel {
    PhaseBlock,
    ElemBlock,
    WarpCentric,
  } kernel = ResampleKernel::ElemBlock;

  // The WarpCentric kernel currently uses cg::reduce(), which requires trivially-copyable types.
  if constexpr (std::is_trivially_copyable_v<output_t>) {
    // There are a couple cases where a warp-centric resampler tends to be faster:
    // 1. When we have a small number of output points, handling one or a few points per warp is an effective
    // way to achieve higher occupancy.
    // 2. When we have many filter taps per output point, each thread in the warp will be able to read
    // multiple elements and the warp will tend to achieve coalesced reads. This helps to prevent loop
    // overhead and barrier stalls from dominating.
    if (output_len <= 2048 || max_phase_len > 256) {
      kernel = ResampleKernel::WarpCentric;
    }
  }

  // Currently, we select only ElemBlock or WarpCentric to keep things simpler. However, there are some
  // cases where PhaseBlock is the fastest kernel. If there are specific parameter sets of interest, then
  // we can benchmark the PhaseBlock method and, if it proves fastest, use that method in those cases.

  // Desired number of blocks to reach high occupancy
  constexpr index_t DESIRED_MIN_GRID_SIZE = 8192;
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));
  dim3 grid(num_batches, 1, 1);
  // comp_unit is either a thread or a warp, depending on the kernel. It is the size of the computational
  // unit that collectively computes a single output value.
  auto compute_elems_per_comp_unit = [&grid](index_t max_outlen_per_cta, int cta_comp_unit_count) -> index_t {
    const int start_batch_size = grid.x * grid.y;
    const index_t desired_extra_batches = (DESIRED_MIN_GRID_SIZE + start_batch_size - 1) /
      start_batch_size;
    const index_t max_outlen_per_comp_unit = (max_outlen_per_cta + cta_comp_unit_count - 1) /
      cta_comp_unit_count;
    grid.z = static_cast<uint32_t>(std::min(desired_extra_batches, max_outlen_per_comp_unit));
    return (max_outlen_per_cta + cta_comp_unit_count * grid.z - 1) / (cta_comp_unit_count * grid.z);
  };

  constexpr int THREADS = MATX_RESAMPLE_POLY_MAX_NUM_THREADS;
  if (kernel == ResampleKernel::PhaseBlock) {
    const size_t smemBytes = (sizeof(filter_t) * max_phase_len <= MATX_RESAMPLE_POLY_MAX_SMEM_BYTES) ?
      sizeof(filter_t) * max_phase_len : 0;
    const index_t max_output_len_per_phase = (output_len + up - 1) / up;
    grid.y = static_cast<int>(up);
    const index_t elems_per_thread = compute_elems_per_comp_unit(max_output_len_per_phase, THREADS);
    if (downcast_to_32b_index()) {
      ResamplePoly1D_PhaseBlock<THREADS, OutType, InType, FilterType, int32_t><<<grid, THREADS, smemBytes, stream>>>(
        o, i, filter, static_cast<int32_t>(up), static_cast<int32_t>(down),
        static_cast<int32_t>(elems_per_thread));
    } else {
      ResamplePoly1D_PhaseBlock<THREADS, OutType, InType, FilterType, index_t><<<grid, THREADS, smemBytes, stream>>>(
        o, i, filter, up, down, elems_per_thread);
    }
  } else if (kernel == ResampleKernel::ElemBlock) {
    const size_t filter_sz_bytes = (filter_len % 2 == 0) ? sizeof(filter_t)*(filter_len+1) : sizeof(filter_t)*filter_len;
    const size_t smemBytes = (filter_sz_bytes <= MATX_RESAMPLE_POLY_MAX_SMEM_BYTES) ? filter_sz_bytes : 0;
    const index_t elems_per_thread = compute_elems_per_comp_unit(output_len, THREADS);
    if (downcast_to_32b_index()) {
      ResamplePoly1D_ElemBlock<THREADS, OutType, InType, FilterType, int32_t><<<grid, THREADS, smemBytes, stream>>>(
        o, i, filter, static_cast<int32_t>(up), static_cast<int32_t>(down),
        static_cast<int32_t>(elems_per_thread));
    } else {
      ResamplePoly1D_ElemBlock<THREADS, OutType, InType, FilterType, index_t><<<grid, THREADS, smemBytes, stream>>>(
        o, i, filter, up, down, elems_per_thread);
    }
  } else {
    // We only select the WarpCentric kernel for trivially copyable types, but we need this
    // constexpr if to avoid instantiating the kernel with inappropriate types.
    if constexpr (std::is_trivially_copyable_v<output_t>) {
      const size_t filter_sz_bytes = (filter_len % 2 == 0) ? sizeof(filter_t)*(filter_len+1) : sizeof(filter_t)*filter_len;
      const size_t smemBytes = (filter_sz_bytes <= MATX_RESAMPLE_POLY_MAX_SMEM_BYTES) ? filter_sz_bytes : 0;
      static_assert(THREADS % WARP_SIZE == 0);
      const index_t elems_per_warp = compute_elems_per_comp_unit(output_len, THREADS/WARP_SIZE);
      if (downcast_to_32b_index()) {
        ResamplePoly1D_WarpCentric<THREADS, OutType, InType, FilterType, int32_t><<<grid, THREADS, smemBytes, stream>>>(
          o, i, filter, static_cast<int32_t>(up), static_cast<int32_t>(down),
          static_cast<int32_t>(elems_per_warp));
      } else {
        ResamplePoly1D_WarpCentric<THREADS, OutType, InType, FilterType, index_t><<<grid, THREADS, smemBytes, stream>>>(
          o, i, filter, up, down, elems_per_warp);
      }
    }
  }
#endif
}

} // end namespace detail


/**
 * @brief 1D polyphase resampler
 * 
 * @tparam OutType Type of output
 * @tparam InType Type of input
 * @tparam FilterType Type of filter
 * @param out Output tensor
 * @param in Input operator
 * @param f Filter operator
 * @param up Factor by which to upsample
 * @param down Factor by which to downsample
 * @param stream CUDA stream on which to run the kernel(s)
 */
template <typename OutType, typename InType, typename FilterType>
inline void resample_poly_impl(OutType &out, const InType &in, const FilterType &f,
                   index_t up, index_t down, cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  constexpr int RANK = InType::Rank();

  MATX_STATIC_ASSERT(OutType::Rank() == InType::Rank(), matxInvalidDim);
  // Currently only support 1D filters.
  MATX_STATIC_ASSERT(FilterType::Rank() == 1, matxInvalidDim);

  MATX_ASSERT_STR(up > 0, matxInvalidParameter, "up must be positive");
  MATX_ASSERT_STR(down > 0, matxInvalidParameter, "down must be positive");

  for(int i = 0 ; i < RANK-1; i++) {
    MATX_ASSERT_STR(out.Size(i) == in.Size(i), matxInvalidDim, "resample_poly: input/output must have matched batch sizes");
  }

  [[maybe_unused]] const index_t up_size = in.Size(RANK-1) * up;
  [[maybe_unused]] const index_t outlen = up_size / down + ((up_size % down) ? 1 : 0);
  MATX_ASSERT_STR(out.Size(RANK-1) == outlen, matxInvalidDim, "resample_poly: output size mismatch");

  const index_t g = std::gcd(up, down);
  up /= g;
  down /= g;

  // There are two ways to interpret resampling when up == down == 1. One is
  // that it is a no-op and thus we should just return a copy of the input
  // tensor. Another is that polyphase resampling is logically equivalent to
  // upsampling, convolving with a filter kernel, and then downsampling, in
  // which case up == down == 1 is equivalent to convolution. We apply the
  // first interpretation and return a copy of the input tensor. This matches
  // the behavior of scipy.
  if (up == 1 && down == 1) {
    (out = in).run(stream);
    return;
  }

  matxResamplePoly1DInternal(out, in, f, up, down, stream);
}

} // end namespace matx
