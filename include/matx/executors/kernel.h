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

#include "matx/core/error.h"
#include "matx/core/get_grid_dims.h"

namespace matx {
namespace detail {

#ifdef __CUDACC__  
template <class Op> __global__ void matxOpT0Kernel(Op op) { 
  if constexpr (std::is_pointer_v<Op>) {
    (*op)(); 
  }
  else {
    op();
  }
}

template <class Op>
__global__ void matxOpT1Kernel(Op op, index_t size0) {
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size0) {
    if constexpr (std::is_pointer_v<Op>) {
      (*op)(idx); 
    }
    else {
      op(idx);
    }
  }
}

template <class Op>
__global__ void matxOpT2Kernel(Op op, index_t size0, index_t size1) {
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (idx < size1 && idy < size0) {
    if constexpr (std::is_pointer_v<Op>) {
      (*op)(idy, idx); 
    }
    else {
      op(idy, idx);
    }    
  }
}

template <class Op>
__global__ void matxOpT3Kernel(Op op, index_t size0, index_t size1, index_t size2) {
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  index_t idz = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
  if (idx < size2 && idy < size1 && idz < size0) {
    if constexpr (std::is_pointer_v<Op>) {
      (*op)(idz, idy, idx); 
    }
    else {
      op(idz, idy, idx);
    }      
  }
}

template <class Op>
__global__ void matxOpT4Kernel(Op op, index_t size0, index_t size1, index_t size2, index_t size3) {
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  index_t idy = nmy % size2;
  index_t idz = nmy / size2;
  index_t idw = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
  if (idx < size3 && idy < size2 && idz < size1 && idw < size0) {
    if constexpr (std::is_pointer_v<Op>) {
      (*op)(idw, idz, idy, idx); 
    }
    else {
      op(idw, idz, idy, idx);
    }      
  }
}

/**
 * @brief Launch an operator with rank N
 * 
 * @tparam Op operator type
 * @param op operator
 * @param sizes sizes of each dimension
 * @param mult Product of sizes of all but first dimension
 */
template <class Op>
__global__ void matxOpTDKernel(Op op, const std::array<index_t, Op::Rank()> sizes, index_t mult) {
  std::array<index_t, Op::Rank()> indices;
  
  // Compute the index into the operator for this thread. N-D tensors require more computations
  // since we're limited to 3 dimensions in both grid and block, so we need to iterate to compute
  // our index.
  index_t x_abs = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool valid = x_abs < mult*sizes[0];
  #pragma unroll
  for (int r = 0; r < Op::Rank(); r++) {
    indices[r] = x_abs / mult;
    x_abs -= indices[r] * mult;      
    mult /= sizes[r+1]; 
  }

  if (valid) {
    if constexpr (std::is_pointer_v<Op>) {
      (*op)(indices); 
    }
    else {
      op(indices);
    }      
  }
}
#endif
}

constexpr int CUDA_MAX_VAL_PARAM = 4096; ///< Parameter size limit for single kernel

} // end namespace matx
