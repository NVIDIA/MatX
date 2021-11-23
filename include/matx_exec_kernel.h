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

constexpr int CUDA_MAX_VAL_PARAM = 4096;

namespace matx {

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
#endif

/**
 * @brief Executes operators on the host on a CUDA-enabled device
 * 
 * Optionally takes a stream for asynchronous execution
 * 
 */
class CUDADeviceExecutor {
  public:
    using matx_executor = bool;
    CUDADeviceExecutor(cudaStream_t stream) : stream_(stream) {}
    CUDADeviceExecutor() : stream_(0) {}

    template <typename Op>
    void Exec(Op &op) const noexcept {
  #ifdef __CUDACC__      
      dim3 threads, blocks;  
      
      // Parameters passed by value in CUDA are limited to 4096B. If the user exceeds this, we 
      // need to error out and have them break up the statement
      MATX_STATIC_ASSERT((sizeof(op) + sizeof(index_t) * Op::Rank()) <= CUDA_MAX_VAL_PARAM, 
        "Parameter buffer to device is limited to 4096B. Please break up your operator statement into multiple executions to limit the size of the parameters");

      if constexpr (op.Rank() == 0) {
        threads = 1;
        blocks = 1;
        matxOpT0Kernel<<<blocks, threads, 0, stream_>>>(op);
      }
      else if constexpr (op.Rank() == 1) {
        index_t size0 = op.Size(0);

        get_grid_dims(blocks, threads, {size0}, 256);
        matxOpT1Kernel<<<blocks, threads, 0, stream_>>>(op, size0);      
      }
      else if constexpr (op.Rank() == 2) {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);

        get_grid_dims(blocks, threads, {size0, size1}, 256);
        matxOpT2Kernel<<<blocks, threads, 0, stream_>>>(op, size0, size1);
      }
      else if constexpr (op.Rank() == 3) {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);
        index_t size2 = op.Size(2);

        get_grid_dims(blocks, threads, {size0, size1, size2}, 256);
        matxOpT3Kernel<<<blocks, threads, 0, stream_>>>(op, size0, size1, size2);
      }
      else {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);
        index_t size2 = op.Size(2);
        index_t size3 = op.Size(3);

        get_grid_dims(blocks, threads, {size0, size1, size2, size3}, 256);
        matxOpT4Kernel<<<blocks, threads, 0, stream_>>>(op, size0, size1, size2, size3);
      } 
  #else
      MATX_THROW(matxNotSupported, "Cannot execute device function from host compiler");    
  #endif    
    }

  private:
    cudaStream_t stream_;
};


} // end namespace matx
