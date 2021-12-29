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

/**
 * @brief Executes operators on the host on a CUDA-enabled device
 * 
 * Optionally takes a stream for asynchronous execution
 * 
 */
class CUDADeviceExecutor {
  public:
    using matx_executor = bool; ///< Type trait indicating this is an executor
    /**
     * @brief Construct a new CUDADeviceExecutor with a stream
     * 
     * @param stream CUDA stream
     */
    CUDADeviceExecutor(cudaStream_t stream) : stream_(stream) {}

    /**
     * @brief Construct a new CUDADeviceExecutor object using the default stream
     * 
     */
    CUDADeviceExecutor() : stream_(0) {}

    /**
     * Execute an operator on a device
     * 
     * @tparam Op Operator type
     * @param op value
     **/
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
        detail::matxOpT0Kernel<<<blocks, threads, 0, stream_>>>(op);
      }
      else {
        std::array<index_t, op.Rank()> sizes;
        for (int i = 0; i < op.Rank(); i++) {
          sizes[i] = op.Size(i);
        }        

        detail::get_grid_dims<op.Rank()>(blocks, threads, sizes, 256);

        if constexpr (op.Rank() == 1) {
          detail::matxOpT1Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0]);      
        }
        else if constexpr (op.Rank() == 2) {
          detail::matxOpT2Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
        }
        else if constexpr (op.Rank() == 3) {
          detail::matxOpT3Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
        }
        else if constexpr (op.Rank() == 4) {
          detail::matxOpT4Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
        }        
        else {
          index_t dims = std::accumulate(std::begin(sizes) + 1, std::end(sizes), 1, std::multiplies<index_t>());
          detail::matxOpTDKernel<<<blocks, threads, 0, stream_>>>(op, sizes, dims);
        } 
      }
  #else
      MATX_THROW(matxNotSupported, "Cannot execute device function from host compiler");    
  #endif    
    }

  private:
    cudaStream_t stream_;
};

} // end namespace matx
