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
#include <type_traits>

#include "matx_error.h"
#include "matx_get_grid_dims.h"

namespace matx {

template <class Op> __global__ void matxOpT0Kernel(Op op) { op(); }

template <class Op>
__launch_bounds__(256) __global__ void matxOpT1Kernel(Op op, index_t size)
{
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    op(idx);
  }
}

template <class Op>
__launch_bounds__(256) __global__
    void matxOpT2Kernel(Op op, index_t size0, index_t size1)
{
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (idx < size1 && idy < size0) {
    op(idy, idx);
  }
}

template <class Op>
__launch_bounds__(256) __global__
    void matxOpT3Kernel(Op op, index_t size0, index_t size1, index_t size2)
{
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  index_t idz = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
  if (idx < size2 && idy < size1 && idz < size0) {
    op(idz, idy, idx);
  }
}

template <class Op>
__launch_bounds__(256) __global__
    void matxOpT4Kernel(Op op, index_t size0, index_t size1, index_t size2,
                        index_t size3)
{
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  index_t idy = nmy % size2;
  index_t idz = nmy / size2;
  index_t idw = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
  if (idx < size3 && idy < size2 && idz < size1 && idw < size0) {
    op(idw, idz, idy, idx);
  }
}

template <class Op> void exec(Op op, cudaStream_t stream = 0)
{
  dim3 threads, blocks;

  if constexpr (op.Rank() == 0) {
    threads = 1;
    blocks = 1;

    matxOpT0Kernel<<<blocks, threads, 0, stream>>>(op);
  }
  else if constexpr (op.Rank() == 1) {
    index_t size0 = op.Size(0);

    get_grid_dims(blocks, threads, size0, 256);
    matxOpT1Kernel<<<blocks, threads, 0, stream>>>(op, size0);
  }
  else if constexpr (op.Rank() == 2) {
    index_t size0 = op.Size(0);
    index_t size1 = op.Size(1);

    get_grid_dims(blocks, threads, size0, size1, 256);
    matxOpT2Kernel<<<blocks, threads, 0, stream>>>(op, size0, size1);
  }
  else if constexpr (op.Rank() == 3) {
    index_t size0 = op.Size(0);
    index_t size1 = op.Size(1);
    index_t size2 = op.Size(2);

    get_grid_dims(blocks, threads, size0, size1, size2, 256);
    matxOpT3Kernel<<<blocks, threads, 0, stream>>>(op, size0, size1, size2);
  }
  else if constexpr (op.Rank() == 4) {

    index_t size0 = op.Size(0);
    index_t size1 = op.Size(1);
    index_t size2 = op.Size(2);
    index_t size3 = op.Size(3);

    get_grid_dims(blocks, threads, size0, size1, size2, size3, 256);
    matxOpT4Kernel<<<blocks, threads, 0, stream>>>(op, size0, size1, size2,
                                                   size3);
  }
}
} // end namespace matx
