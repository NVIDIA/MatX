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
#include "matx/kernels/channelize_poly.cuh"
#include "matx/operators/fft.h"
#include "matx/operators/slice.h"

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

template <typename OutType, typename InType, typename FilterType>
inline void matxChannelizePoly1DInternal(OutType o, const InType &i,
                                     const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  
  index_t filter_len = filter.Size(FilterType::Rank()-1);

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
  const int num_batches = static_cast<int>(TotalSize(i)/i.Size(i.Rank() - 1));

  const int THREADS = 256;
  const index_t ELTS_PER_THREAD = CHANNELIZE_POLY1D_ELEMS_PER_THREAD * THREADS;
  const int elem_blocks = static_cast<int>(
    (nout_per_channel + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD);
  dim3 grid(elem_blocks, static_cast<int>(num_channels), num_batches);
  ChannelizePoly1D<THREADS, OutType, InType, FilterType><<<grid, THREADS, 0, stream>>>(
      o, i, filter);
#endif
}

template <typename OutType, typename InType, typename FilterType>
inline size_t matxChannelizePoly1DInternal_SmemSizeBytes(const OutType &o, const InType &, const FilterType &filter)
{
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;

  index_t filter_len = filter.Size(FilterType::Rank()-1);

  const index_t num_channels = o.Size(OutType::Rank()-1);
  const index_t nout_per_channel = o.Size(OutType::Rank()-2);
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

template <typename OutType, typename InType, typename FilterType>
inline void matxChannelizePoly1DInternal_Smem(OutType o, const InType &i, const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;

  index_t filter_len = filter.Size(FilterType::Rank()-1);

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
  ChannelizePoly1D_Smem<OutType, InType, FilterType><<<grid, block, smem_size, stream>>>(
      o, i, filter, elem_per_block);
#endif
}

template <typename OutType, typename InType, typename FilterType>
inline void matxChannelizePoly1DInternal_FusedChan(OutType o, const InType &i,
                                     const FilterType &filter, cudaStream_t stream)
{
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  
  index_t filter_len = filter.Size(FilterType::Rank()-1);

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
      ChannelizePoly1D_FusedChan<THREADS, 2, OutType, InType, FilterType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 3:
      ChannelizePoly1D_FusedChan<THREADS, 3, OutType, InType, FilterType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 4:
      ChannelizePoly1D_FusedChan<THREADS, 4, OutType, InType, FilterType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 5:
      ChannelizePoly1D_FusedChan<THREADS, 5, OutType, InType, FilterType><<<grid,THREADS,0,stream>>>(o, i, filter);
      break;
    case 6:
      ChannelizePoly1D_FusedChan<THREADS, 6, OutType, InType, FilterType><<<grid,THREADS,0,stream>>>(o, i, filter);
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
 * constituent channels, each corresponding to a band of the input signal bandwidth. The current
 * implementation only supports maximally decimated (i.e., critically sampled) channelizers wherein the
 * decimation factor is equivalent to the number of channels and the channels are non-overlapping.
 * 
 * @tparam OutType Type of output
 * @tparam InType Type of input
 * @tparam FilterType Type of filter
 * @param out Output tensor
 * @param in Input operator
 * @param f Filter operator
 * @param num_channels Number of channels in which to separate the signal. Must be greater than 1.
 * @param decimation_factor Factor by which to downsample the input signal into the channels. Currently,
 * the only supported value of decimation_factor is a value equal to num_channels. This corresponds to
 * the maximally decimated, or critically sampled, case. It is also possible for decimation_factor to
 * be less than num_channels, which corresponds to an oversampled case with overlapping channels, but
 * this implementation does not yet support oversampled cases.
 * @param stream CUDA stream on which to run the kernel(s)
 */
template <typename OutType, typename InType, typename FilterType>
inline void channelize_poly_impl(OutType out, const InType &in, const FilterType &f,
                   index_t num_channels, [[maybe_unused]] index_t decimation_factor, cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using input_t = typename InType::value_type;
  using filter_t = typename FilterType::value_type;
  using output_t = typename OutType::value_type;

  constexpr int IN_RANK = InType::Rank();
  constexpr int OUT_RANK = OutType::Rank();

  // The last dimension of the input becomes [num_channels, num_elem_per_channel] in the last
  // two dimensions of the output
  MATX_STATIC_ASSERT_STR(OUT_RANK == IN_RANK+1, matxInvalidDim, "channelize_poly: output rank should be 1 higher than input");

  MATX_STATIC_ASSERT_STR(is_complex_v<output_t> || is_complex_half_v<output_t>,
    matxInvalidType, "channelize_poly: output type should be complex");

  // Currently only support 1D filters.
  MATX_STATIC_ASSERT_STR(FilterType::Rank() == 1, matxInvalidDim, "channelize_poly: currently only support 1D filters");

  MATX_ASSERT_STR(num_channels > 1, matxInvalidParameter,
    "channelize_poly: num_channels must be greater than 1");
  MATX_ASSERT_STR(decimation_factor > 0, matxInvalidParameter,
    "channelize_poly: decimation_factor must be positive");
  MATX_ASSERT_STR(num_channels == decimation_factor, matxInvalidParameter,
    "channelize_poly: currently only support decimation_factor == num_channels");

  for(int i = 0 ; i < IN_RANK-1; i++) {
    MATX_ASSERT_STR(out.Size(i) == in.Size(i), matxInvalidDim, "channelize_poly: input/output must have matched batch sizes");
  }

  const index_t num_elem_per_channel = (in.Size(IN_RANK-1) + num_channels - 1) / num_channels;

  MATX_ASSERT_STR(out.Size(OUT_RANK-1) == num_channels, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-1 mismatch");
  MATX_ASSERT_STR(out.Size(OUT_RANK-2) == num_elem_per_channel, matxInvalidDim,
    "channelize_poly: output size OUT_RANK-2 mismatch");

  // If neither the input nor the filter is complex, then the filtered samples will be real-valued
  // and we will use an R2C transform. Otherwise, we will use a C2C transform.
  if constexpr (! is_complex_v<input_t> && ! is_complex_half_v<input_t> && ! is_complex_v<filter_t> && ! is_complex_half_v<filter_t>) {
    if (num_channels <= detail::MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD) {
      matxChannelizePoly1DInternal_FusedChan(out, in, f, stream);
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

      if (matxChannelizePoly1DInternal_ShouldUseSmemKernel(out, in, f)) {
        matxChannelizePoly1DInternal_Smem(fft_in_slice, in, f, stream);
      } else {
        matxChannelizePoly1DInternal(fft_in_slice, in, f, stream);
      }
      stop_dims[OUT_RANK-1] = (num_channels/2) + 1;
      auto out_packed = slice<OUT_RANK>(out, start_dims, stop_dims);
      (out_packed = fft(fft_in_slice, num_channels)).run(stream);
      matxChannelizePoly1DUnpackInternal(out, stream);
    }
  } else {
    if (num_channels <= detail::MATX_CHANNELIZE_POLY1D_FUSED_CHAN_KERNEL_THRESHOLD) {
      matxChannelizePoly1DInternal_FusedChan(out, in, f, stream);
    } else {
      if (matxChannelizePoly1DInternal_ShouldUseSmemKernel(out, in, f)) {
        matxChannelizePoly1DInternal_Smem(out, in, f, stream);
      } else {
        matxChannelizePoly1DInternal(out, in, f, stream);
      }
      // Specify FORWARD here to prevent any normalization after the ifft. We do not
      // want any extra scaling on the output values.
      (out = ifft(out, num_channels, FFTNorm::FORWARD)).run(stream);
    }
  }
}
} // end namespace matx
