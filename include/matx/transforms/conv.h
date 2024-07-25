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
#include "matx/kernels/conv.cuh"

namespace matx {
namespace detail {

template <typename OutputType, typename InType, typename FilterType, typename Executor>
inline void matxFFTConv1DInternal(OutputType &o, const InType &i,
                                     const FilterType &filter, matxConvCorrMode_t mode,
                                     const Executor &exec)
{
  const index_t padded_size = i.Size(InType::Rank() - 1) + filter.Size(InType::Rank() - 1) - 1;
  auto in_shape_padded = Shape(i);
  in_shape_padded[InType::Rank() - 1] = padded_size;
  const auto filter_size = filter.Size(FilterType::Rank() - 1);

  index_t slice_start[InType::Rank()];
  index_t slice_end[InType::Rank()];

  std::fill(std::begin(slice_start), std::end(slice_start), 0);
  std::fill(std::begin(slice_end), std::end(slice_end), matxEnd);
  
  auto allocate_tensor = [&](auto shape) {
    if constexpr (is_cuda_executor_v<Executor>) {
      return make_tensor<complex_from_scalar_t<typename InType::value_type>>(shape, MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
    } else {
      return make_tensor<complex_from_scalar_t<typename InType::value_type>>(shape, MATX_HOST_MALLOC_MEMORY);
    }
  };

  auto s1 = allocate_tensor(in_shape_padded);
  auto s2 = allocate_tensor(in_shape_padded);
  auto sifft = allocate_tensor(in_shape_padded);

  if constexpr (! is_complex_v<typename InType::value_type>) {
    slice_end[InType::Rank() - 1] = padded_size/2 + 1;
  }
  auto s1s = slice(s1, slice_start, slice_end);
  if constexpr (! is_complex_v<typename FilterType::value_type>) {
    slice_end[FilterType::Rank() - 1] = padded_size/2 + 1;
  }
  auto s2s = slice(s2, slice_start, slice_end);
  (s1s = fft(i, padded_size)).run(exec);
  (s2s = fft(filter, padded_size)).run(exec);

  // If this is real-valued input we need to accomodate cuFFT's output of N/2+1 complex
  // samples and use r2c to convert back to N.
  if constexpr (!is_complex_v<typename InType::value_type>) {
    slice_end[InType::Rank() - 1] = padded_size / 2 + 1;
    if constexpr (!is_complex_v<typename FilterType::value_type>) {
      (sifft = r2c(slice(s1, slice_start, slice_end) * slice(s2, slice_start, slice_end), padded_size)).
            run(exec);
    }
    else {
      (sifft = r2c(slice(s1, slice_start, slice_end), padded_size) * s2).run(exec);
    }
  }
  else {
    if constexpr (!is_complex_v<typename FilterType::value_type>) {
      (sifft = s1 * r2c(slice(s2, slice_start, slice_end), padded_size)).run(exec);
    }
    else {
      (sifft = s1 * s2).run(exec);
    }
  }

  slice_end[InType::Rank() - 1] = matxEnd;

  // At this point our two signals are complex, regardless of what the input was

  // Write directly to output in FULL mode.
  if (mode == MATX_C_MODE_FULL) {
    if constexpr (is_complex_v<typename InType::value_type> || is_complex_v<typename FilterType::value_type>) {
      (o = ifft(sifft)).run(exec);
    }
    else {
      (o = real(ifft(sifft))).run(exec);
    }
  }
  else if (mode == MATX_C_MODE_SAME) {
    (sifft = ifft(sifft)).run(exec);
    if (filter_size & 1) {
      slice_start[InType::Rank() - 1] = (filter_size - 1) / 2;
    }
    else {
      slice_start[InType::Rank() - 1] = filter_size / 2 - 1;
    }

    slice_end[InType::Rank() - 1] = padded_size - filter_size / 2;

    if constexpr (is_complex_v<typename InType::value_type> || is_complex_v<typename FilterType::value_type>) {
      (o = slice(sifft, slice_start, slice_end)).run(exec);
    }
    else {
      (o = slice(real(sifft), slice_start, slice_end)).run(exec);
    }
  }
  else if (mode == MATX_C_MODE_VALID) {
    (sifft = ifft(sifft)).run(exec);
    slice_start[InType::Rank() - 1] = filter_size - 1;
    slice_end[InType::Rank() - 1]   = padded_size - filter_size + 1;

    if constexpr (is_complex_v<typename InType::value_type> || is_complex_v<typename FilterType::value_type>) {
      (o = slice(sifft, slice_start, slice_end)).run(exec);
    }
    else {
      (o = slice(real(sifft), slice_start, slice_end)).run(exec);
    }
  }
}


template <typename OutputType, typename InType, typename FilterType>
inline void matxDirectConv1DInternal(OutputType &o, const InType &i,
                                     const FilterType &filter, matxConvCorrMode_t mode,
                                     const cudaExecutor &exec)
{
  MATX_STATIC_ASSERT(OutputType::Rank() == InType::Rank(), matxInvalidDim);

  MATX_ASSERT_STR(mode != MATX_C_MODE_FULL || o.Size(o.Rank()-1) == i.Size(i.Rank()-1) + filter.Size(filter.Rank()-1) - 1,
      matxInvalidSize, "Output size for FULL convolution incorrect");
  MATX_ASSERT_STR(mode != MATX_C_MODE_SAME || o.Size(o.Rank()-1) == i.Size(i.Rank()-1),
      matxInvalidSize, "Output size for SAME convolution incorrect");

#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const auto stream = exec.getStream();

  using strip_input_t = typename InType::value_type;
  using strip_filter_t = typename FilterType::value_type;
  using shape_type = std::conditional_t<has_shape_type_v<OutputType>, typename OutputType::shape_type, index_t>;

  size_t filter_len = filter.Size(filter.Rank()-1);

  size_t filter_shm = sizeof(strip_filter_t) * filter_len;
  size_t signal_shm = sizeof(strip_input_t) * (CONV1D_ELEMENTS_PER_BLOCK + filter_len);

  // align filter size to signal size
  size_t align = std::alignment_of_v<InType>;
  filter_shm = (filter_shm + align - 1) / align * align;

  size_t shmsize = filter_shm + signal_shm;

  shape_type sig_len = i.Size(OutputType::Rank() - 1);
  int work_per_block = CONV1D_ELEMENTS_PER_BLOCK;
  unsigned int num_blocks = (unsigned int)(sig_len + filter.Size(filter.Rank()-1) + work_per_block -1) / work_per_block;

  // number below was chosen arbitrarily.  Cannot be more than 65536.
  num_blocks = cuda::std::min(num_blocks, 10000U);

  unsigned int grid_size = static_cast<unsigned int>(TotalSize(i)/i.Size(i.Rank() - 1));

  dim3 gsize(grid_size, num_blocks);
  constexpr int EPT = 4;
  constexpr int THREADS = CONV1D_ELEMENTS_PER_BLOCK / EPT;
  static_assert(CONV1D_ELEMENTS_PER_BLOCK % EPT == 0);

  Conv1D<THREADS, EPT><<<gsize, THREADS, shmsize, stream>>>(
      o, i, filter, sig_len, mode);

#endif
}


template <typename OutputType, typename In1Type, typename In2Type>
void matxDirectConv2DInternal(OutputType &o, In1Type &in1,
                              In2Type &in2, matxConvCorrMode_t mode,
                              cudaStream_t stream)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  MATX_STATIC_ASSERT(OutputType::Rank() == In1Type::Rank(), matxInvalidDim);
  MATX_STATIC_ASSERT(OutputType::Rank() == In2Type::Rank(), matxInvalidDim);
  MATX_STATIC_ASSERT(OutputType::Rank() >= 2, matxInvalidDim);

#ifdef __CUDACC__
  constexpr int Rank = OutputType::Rank();

  // TODO dispatch different sizes based on filter size?
  const int BLOCK_X = 16;
  const int BLOCK_Y = 8;
  const int FILTER_SHARED_X = 16;
  const int FILTER_SHARED_Y = 16;
  const int FILTER_REG_X = 4;
  const int FILTER_REG_Y = 8;
  const int ILPY = 8;

  dim3 threads(BLOCK_X, BLOCK_Y, 1);
  int num_batch = int(TotalSize(o) / o.Size(Rank-1) / (o.Size(Rank-2)));
  dim3 blocks( int( (o.Size(Rank-1) + threads.x - 1 ) / threads.x),
               int((o.Size(Rank-2) + (threads.y * ILPY) - 1 ) / (threads.y * ILPY)),
               num_batch);

  Conv2D<OutputType, In1Type, In2Type, BLOCK_X, BLOCK_Y, FILTER_SHARED_X, FILTER_SHARED_Y, FILTER_REG_X, FILTER_REG_Y, ILPY><<<blocks, threads, 0, stream>>>(o, in1, in2, mode, num_batch);
#endif
}
} // end namespace detail

template <typename OutputType, typename In1Type, typename In2Type, typename Executor>
inline void conv1d_impl_internal(OutputType &o, const In1Type &i1, const In2Type &i2,
                   matxConvCorrMode_t mode, matxConvCorrMethod_t method, const Executor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(In1Type::Rank() == In2Type::Rank());

  if (mode == MATX_C_MODE_SAME) {
    MATX_ASSERT_STR(o.Size(OutputType::Rank() - 1) == cuda::std::max(i1.Size(i1.Rank()-1), i2.Size(i2.Rank()-1)), matxInvalidSize,
      "Output size for SAME mode convolution must match largest input size");
  }

  if (mode == MATX_C_MODE_VALID) {
    MATX_ASSERT_STR(o.Size(OutputType::Rank() - 1) ==
      cuda::std::max(i1.Size(i1.Rank()-1), i2.Size(i2.Rank()-1)) - cuda::std::min(i1.Size(i1.Rank()-1), i2.Size(i2.Rank()-1)) + 1, matxInvalidSize,
      "Output size for VALID mode convolution must be N - L + 1");
  }

  const int Rank = In1Type::Rank();
  //detail::tensor_impl_t<typename OutputType::value_type, OutputType::Rank(), typename OutputType::desc_type> &o_base = o;
  typename detail::base_type<OutputType>::type &o_base = o;
  const typename detail::base_type<In1Type>::type &in1_base = i1;
  const typename detail::base_type<In2Type>::type &in2_base = i2;

  if (i1.Size(Rank-1) < i2.Size(Rank-1)) {
    if (method == MATX_C_METHOD_DIRECT) {
      if constexpr (detail::CheckDirect1DConvSupport<Executor>()) {
        detail::matxDirectConv1DInternal(o_base, in2_base, in1_base, mode, exec);
      } else {
        MATX_THROW(matxNotSupported, "direct conv1d() only supports the CUDA executor currently");
      }
    }
    else {
      detail::matxFFTConv1DInternal(o_base, i2, i1, mode, exec);
    }
  }
  else {
    if (method == MATX_C_METHOD_DIRECT) {
      if constexpr (detail::CheckDirect1DConvSupport<Executor>()) {
        detail::matxDirectConv1DInternal(o_base, in1_base, in2_base, mode, exec);
      } else {
        MATX_THROW(matxNotSupported, "direct conv1d() only supports the CUDA executor currently");
      }
    }
    else {
      detail::matxFFTConv1DInternal(o_base, i1, i2, mode, exec);
    }
  }
}


/**
 * @brief 1D convolution
 *
 * @tparam OutputType Type of output
 * @tparam In1Type Type of first input
 * @tparam In2Type Type of second input
 * @param o Output tensor
 * @param i1 First input operator
 * @param i2 Second input operator
 * @param mode Convolution mode
 * @param method Convolution method
 * @param exec Executor
 */
template <typename OutputType, typename In1Type, typename In2Type, typename Executor>
inline void conv1d_impl(OutputType o, const In1Type &i1, const In2Type &i2,
                   matxConvCorrMode_t mode, matxConvCorrMethod_t method, const Executor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if constexpr ( In1Type::Rank() >  In2Type::Rank() ) {
    // broadcast i2 path.  clone i2 across batches

    constexpr int LRank = In1Type::Rank();
    constexpr int SRank = In2Type::Rank();
    constexpr int DRank = LRank - SRank;

    index_t shape[LRank];

    // copy left-most dimensions from i1
    #pragma unroll
    for(int i = 0; i < DRank; i++) {
      shape[i] = i1.Size(i);
    }

    // set right most dimensions as matxKeepDim
    #pragma unroll
    for(int i = 0; i < SRank; i++) {
      shape[DRank+i] = matxKeepDim;
    }

    // clone i2
    auto ci2 = clone<LRank>(i2, shape);

    static_assert(In1Type::Rank() == decltype(ci2)::Rank());

    conv1d_impl_internal(o, i1, ci2, mode, method, exec);

  }  else if constexpr ( In2Type::Rank() >  In1Type::Rank()) {
    // broadcast i1 path.  clone i1 across batches

    constexpr int LRank = In2Type::Rank();
    constexpr int SRank = In1Type::Rank();
    constexpr int DRank = LRank - SRank;
    index_t shape[LRank];

    // copy left-most dimensions from i2
    #pragma unroll
    for(int i = 0; i < DRank; i++) {
      shape[i] = i2.Size(i);
    }

    // set right most dimensions as matxKeepDim
    #pragma unroll
    for(int i = 0; i < SRank; i++) {
      shape[DRank+i] = matxKeepDim;
    }

    // clone i1
    auto ci1 = clone<LRank>(i1, shape);

    static_assert(ci1.Rank() == i2.Rank());

    conv1d_impl_internal(o, ci1, i2, mode, method, exec);

  } else {
    static_assert(In1Type::Rank() == In2Type::Rank());
    // batched pass outer dims must match
    conv1d_impl_internal(o, i1, i2, mode, method, exec);
  }
}


/**
 * @brief 2D convolution
 *
 * @tparam OutputType Type of output
 * @tparam In1Type Type of first input
 * @tparam In2Type Type of second input
 * @param o Output tensor
 * @param in1 First input operator
 * @param in2 Second input operator
 * @param mode Convolution mode
 * @param stream CUDA stream
 */
template <typename OutputType, typename In1Type, typename In2Type>
inline void conv2d_impl(OutputType o, const In1Type in1, const In2Type in2,
                   matxConvCorrMode_t mode, cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  constexpr int Rank1 = In1Type::Rank();
  constexpr int Rank2 = In2Type::Rank();

  if constexpr (In1Type::Rank() == In2Type::Rank()) {
     index_t size1 = in1.Size(Rank1-1) * in1.Size(Rank1-2);
     index_t size2 = in2.Size(Rank2-1) * in2.Size(Rank2-2);
     // smaller size is the filter, set it as second input
     if(size1 >= size2) {
       detail::matxDirectConv2DInternal(o, in1, in2, mode, stream);
     } else {  // swap in1/in2
       detail::matxDirectConv2DInternal(o, in2, in1, mode, stream);
     }
  // These branches clone the inputs to match in rank
  } else if constexpr (In1Type::Rank() <In2Type::Rank()) {
      // in1 is smaller so clone it to match in2
      auto shape = in2.Shape();
      int d = Rank2 - Rank1;
      for(int i = 0; i < Rank1; i++) {
        shape[i+d] = matxKeepDim;
      }
      conv2d_impl(o, clone<Rank2>(in1, shape), in2, mode, stream);
  } else {
      // in1 is smaller so clone it to match in2
      auto shape = in1.Shape();
      int d = Rank1 - Rank2;
      for(int i = 0; i < Rank2; i++) {
        shape[i+d] = matxKeepDim;
      }
      conv2d_impl(o, in1, clone<Rank1>(in2, shape), mode, stream);
  }
}

} // end namespace matx
