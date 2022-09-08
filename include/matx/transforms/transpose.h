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


#include "matx/core/type_utils.h"
#include "matx/kernels/transpose.cuh"

namespace matx
{
  /**
   * Transpose the outer dimensions of a tensor view out-of-place
   *
   * Transposes the two fastest-changing dimensions of a tensor. Any higher
   * dimension is untouched. This has the same effect as permute with {1,0} as the
   * last two dims, but it is much faster for tensors that are already contiguous.
   * For tensors that are not a contiguous view, this function is not allowed.
   *
   * Both tensor views must be the same rank, and the dimensions that moved must
   * match their original size
   *
   * @param out
   *   Tensor to copy into
   * @param in
   *   Tensor to copy from
   * @param stream
   *   CUDA stream to operate in
   *
   */
  template <typename OutputTensor, typename InputTensor>
    __MATX_INLINE__ void transpose(OutputTensor &out,
        const InputTensor &in,
        const cudaStream_t stream)
    {
      constexpr int RANK = OutputTensor::Rank();
      if constexpr (RANK <= 1)
      {
        return;
      }

      if (!in.IsContiguous())
      {
        MATX_THROW(matxInvalidSize, "Must have a linear tensor view for transpose");
      }

#ifdef __CUDACC__  
      size_t shm = sizeof(typename OutputTensor::scalar_type) * TILE_DIM * (TILE_DIM + 1);
      if constexpr (RANK == 2)
      {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid(static_cast<int>((in.Size(RANK - 1) + TILE_DIM - 1) / TILE_DIM),
            static_cast<int>((in.Size(RANK - 2) + TILE_DIM - 1) / TILE_DIM));
        transpose_kernel_oop<<<grid, block, shm, stream>>>(out, in);
      }
      else if constexpr (RANK >= 3)
      {
        int batch_dims =
          static_cast<int>(in.TotalSize() / (in.Size(RANK - 1) * in.Size(RANK - 2)));

        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid(static_cast<int>((in.Size(RANK - 1) + TILE_DIM - 1) / TILE_DIM),
            static_cast<int>((in.Size(RANK - 2) + TILE_DIM - 1) / TILE_DIM),
            batch_dims);
        transpose_kernel_oop<<<grid, block, shm, stream>>>(out, in);
      }
#else
     MATX_THROW(matxNotSupported, "Transpose not supported on host");
#endif    
    };
} // end namespace matx
