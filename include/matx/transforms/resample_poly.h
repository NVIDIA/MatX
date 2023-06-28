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
  
  using input_t = typename InType::scalar_type;
  using filter_t = typename FilterType::scalar_type;
  using shape_type = typename OutType::shape_type;
  
  shape_type filter_len = filter.Size(FilterType::Rank()-1);

  // Even-length filters will be prepended with a single 0 to make them odd-length
  const int max_phase_len = (filter_len % 2 == 0) ?
    static_cast<int>((filter_len + 1 + up - 1) / up) :
    static_cast<int>((filter_len + up - 1) / up);
  const size_t filter_shm = sizeof(filter_t) * max_phase_len;

  const int num_phases = static_cast<int>(up);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));
  dim3 grid(num_batches, num_phases);

  constexpr int THREADS = 128;
  ResamplePoly1D<THREADS, OutType, InType, FilterType><<<grid, THREADS, filter_shm, stream>>>(
      o, i, filter, up, down);

#endif
}

} // end namespace detail

// Simple gcd implementation using the Euclidean algorithm.
// If large number support is needed, or if this function becomes performance
// sensitive, then this implementation may be insufficient. Typically, up/down
// factors for resampling will be known in a signal processing pipeline and
// thus the user would already supply co-prime up/down factors. In that case,
// b will be 0 below after one iteration and this implementation quickly identifies
// the factors as co-prime.
static index_t gcd(index_t a, index_t b) {
  while (b != 0) {
    const index_t t = b;
    b = a % b;
    a = t;
  }
  return a;
};

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
inline void resample_poly(OutType &out, const InType &in, const FilterType &f,
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

  const index_t up_size = in.Size(RANK-1) * up;
  const index_t outlen = up_size / down + ((up_size % down) ? 1 : 0);

  MATX_ASSERT_STR(out.Size(RANK-1) == outlen, matxInvalidDim, "resample_poly: output size mismatch");

  const index_t g = gcd(up, down);
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
