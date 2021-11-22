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

#include "kernels/matx_conv_kernels.cuh"
#include "matx_dim.h"
#include "matx_error.h"
#include "matx_tensor.h"

namespace matx {

template <typename T, int RANK, typename InType, typename FilterType>
inline void matxDirectConv1DInternal(tensor_impl_t<T, RANK> o, InType i,
                                     FilterType filter, matxConvCorrMode_t mode,
                                     cudaStream_t stream)
{
  using strip_input_t = typename InType::scalar_type;
  using strip_filter_t = typename FilterType::scalar_type;
  MATX_STATIC_ASSERT(RANK == InType::Rank(), matxInvalidDim);
  MATX_STATIC_ASSERT(FilterType::Rank() == 1, matxInvalidDim);

#ifdef __CUDACC__  
  // Scale the filter
  size_t filter_shm;
  if (sizeof(strip_filter_t) < sizeof(strip_input_t)) {
    filter_shm = (filter.Size(0) * sizeof(strip_filter_t) + (sizeof(strip_input_t)-1)) / sizeof(strip_input_t) * sizeof(strip_input_t);
  }
  else {
    filter_shm = filter.Size(0) * sizeof(strip_filter_t);
  }

  auto shmsize = filter_shm + sizeof(strip_input_t) * (filter.Size(0) + BLOCK_SIZE_NON_RECURSIVE);

  index_t sig_len = i.Size(RANK - 1);
  float work_per_block =
      static_cast<float>(BLOCK_SIZE_NON_RECURSIVE - filter.Size(0) + 1);
  uint32_t num_blocks = static_cast<uint32_t>(std::ceil(
      static_cast<float>(sig_len + filter.Size(0) - 1) / work_per_block));

  if constexpr (RANK == 1) {
    dim3 gsize(num_blocks, 1);
    Conv1D<<<gsize, BLOCK_SIZE_NON_RECURSIVE, shmsize, stream>>>(
          o, i, filter, sig_len, filter.Size(0), mode);
  }
  else if constexpr (RANK == 2) {
    dim3 gsize(num_blocks, static_cast<int>(i.Size(0)));
    Conv1D<<<gsize, BLOCK_SIZE_NON_RECURSIVE, shmsize, stream>>>(
            o, i, filter, sig_len, filter.Size(0), mode);
  }
  else if constexpr (RANK == 3) {
    dim3 gsize(num_blocks, static_cast<int>(i.Size(1)),
               static_cast<int>(i.Size(0)));
    Conv1D<<<gsize, BLOCK_SIZE_NON_RECURSIVE, shmsize, stream>>>(
            o, i, filter, sig_len, filter.Size(0), mode);
  }
  else {
    static_assert(RANK == 4);
    dim3 gsize(num_blocks, static_cast<int>(i.Size(2)),
               static_cast<int>(i.Size(0) * i.Size(1)));
    Conv1D<<<gsize, BLOCK_SIZE_NON_RECURSIVE, shmsize, stream>>>(
            o, i, filter, sig_len, filter.Size(0), mode);
  }
#endif  
}

// Entry point that allows swappable inputs, and also optimizes shared memory by
// passing in the shortest signal as the filter. Note the swap parameter does
// not do anything for convolution, so we never swap unless it's a correlation
template <typename T, int RANK, typename In1Type, typename In2Type>
inline void conv1d(tensor_t<T, RANK> o, In1Type i1, In2Type i2,
                   matxConvCorrMode_t mode, cudaStream_t stream)
{
  tensor_impl_t<T,RANK> &o_base = o;
  typename base_type<In1Type>::type &in1_base = i1;
  typename base_type<In2Type>::type &in2_base = i2;

  if constexpr (In1Type::Rank() < In2Type::Rank()) {
    matxDirectConv1DInternal(o_base, in2_base, in1_base, mode, stream);
  }
  else if constexpr (In1Type::Rank() == In2Type::Rank()) {
    MATX_STATIC_ASSERT(RANK == 1, matxInvalidDim);
    if (i1.Size(0) < i2.Size(0)) {
      matxDirectConv1DInternal(o_base, in2_base, in1_base, mode, stream);
    }
    else {
      matxDirectConv1DInternal(o_base, in1_base, in2_base, mode, stream);
    }
  }
  else {
    matxDirectConv1DInternal(o_base, in1_base, in2_base, mode, stream);
  }
}

template <typename T, int RANK, typename InType, typename FilterType>
void matxDirectConv2DInternal(tensor_impl_t<T, RANK> &o, InType &i,
                              FilterType &filter, matxConvCorrMode_t mode,
                              cudaStream_t stream)
{
  MATX_STATIC_ASSERT(RANK == InType::Rank(), matxInvalidDim);
  MATX_STATIC_ASSERT(FilterType::Rank() == 2, matxInvalidDim);

  using strip_input_t = typename InType::scalar_type;
  auto shmsize = sizeof(strip_input_t) * filter.Size(0) * filter.Size(1);

#ifdef __CUDACC__  
  if constexpr (RANK == 1) {
    MATX_THROW(matxInvalidDim,
               "matxDirectConv2D not supported on Rank 1 Tensors");
  }
  else if constexpr (RANK == 2) {
    dim3 gsize(
        static_cast<int>(std::ceil(static_cast<double>(o.Size(1)) / 32.0)),
        static_cast<int>(std::ceil(static_cast<double>(o.Size(0)) / 32.0)), 1);
    dim3 bsize(32, 32);
    Conv2D<<<gsize, bsize, shmsize, stream>>>(o, i, filter, mode);
  }
  else if constexpr (RANK == 3) {
    dim3 gsize(static_cast<unsigned int>(
                   std::ceil(static_cast<double>(o.Size(2)) / 32.0)),
               static_cast<unsigned int>(
                   std::ceil(static_cast<double>(o.Size(1)) / 32.0)),
               static_cast<unsigned int>(o.Size(0)));
    dim3 bsize(32, 32);
    Conv2D<<<gsize, bsize, shmsize, stream>>>(o, i, filter, mode);
  }
  else {
    static_assert(RANK == 4);
    dim3 gsize(static_cast<unsigned int>(
                   std::ceil(static_cast<double>(o.Size(2)) / 32.0)),
               static_cast<unsigned int>(
                   std::ceil(static_cast<double>(o.Size(1)) / 32.0)),
               static_cast<unsigned int>(o.Size(0) * o.Size(1)));
    dim3 bsize(32, 32);
    Conv2D<<<gsize, bsize, shmsize, stream>>>(o, i, filter, mode);
  }
#endif  
}

// Entry point that allows swappable inputs, and also optimizes shared memory by
// passing in the shortest signal as the filter
template <typename T, int RANK, typename In1Type, typename In2Type>
inline void conv2d(tensor_t<T, RANK> o, In1Type i1, In2Type i2,
                   matxConvCorrMode_t mode, cudaStream_t stream)
{
  tensor_impl_t<T,RANK> &o_base = o;
  typename base_type<In1Type>::type &in1_base = i1;
  typename base_type<In2Type>::type &in2_base = i2;

  if constexpr (In1Type::Rank() < In2Type::Rank()) {
    matxDirectConv2DInternal(o_base, in2_base, in1_base, mode, stream);
  }
  else if constexpr (In1Type::Rank() == In2Type::Rank()) {
    MATX_STATIC_ASSERT(RANK == 2, matxInvalidDim);
    if (i1.Size(0) * i1.Size(1) < i2.Size(0) * i2.Size(1)) {
      matxDirectConv2DInternal(o_base, in2_base, in1_base, mode, stream);
    }
    else {
      matxDirectConv2DInternal(o_base, in1_base, in2_base, mode, stream);
    }
  }
  else {
    matxDirectConv2DInternal(o_base, in1_base, in2_base, mode, stream);
  }
}

} // end namespace matx
