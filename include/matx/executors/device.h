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

#include "matx/core/defines.h"
#include "matx/executors/host.h"
#include "matx/executors/kernel.h"

namespace matx
{

  /**
   * @brief Executes operators on the host on a CUDA-enabled device
   * 
   * Optionally takes a stream for asynchronous execution
   * 
   */
  class cudaExecutor {
    public:
      using matx_cuda = bool;  // signal this is a GPU executor
      using matx_executor = bool; ///< Type trait indicating this is an executor
      /**
       * @brief Construct a new cudaExecutor with a stream
       * 
       * @param stream CUDA stream
       */
      cudaExecutor(cudaStream_t stream) : stream_(stream) {}
      cudaExecutor(int stream) : stream_(reinterpret_cast<cudaStream_t>(stream)) {}

      /**
       * @brief Construct a new cudaExecutor object using the default stream
       * 
       */
      cudaExecutor() : stream_(0) {}

      /*
       * @breif Returns stream associated with executor
       */
      auto getStream() { return stream_; }

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

          if constexpr (Op::Rank() == 0) {
            threads = 1;
            blocks = 1;
            detail::matxOpT0Kernel<<<blocks, threads, 0, stream_>>>(op);
          }
          else {
            std::array<index_t, Op::Rank()> sizes;
            for (int i = 0; i < Op::Rank(); i++) {
              sizes[i] = op.Size(i);
            }        

            bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, 256);

            if constexpr (Op::Rank() == 1) {
              if(stride) {
                detail::matxOpT1StrideKernel<<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else {
                detail::matxOpT1Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              }
            }
            else if constexpr (Op::Rank() == 2) {
              if(stride) {
                detail::matxOpT2StrideKernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
              } else {
                detail::matxOpT2Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
              }
            }
            else if constexpr (Op::Rank() == 3) {
              if(stride) {
                detail::matxOpT3StrideKernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
              } else {
                detail::matxOpT3Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
              }
            }
            else if constexpr (Op::Rank() == 4) {
              if(stride) {
                detail::matxOpT4StrideKernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
              } else {
                detail::matxOpT4Kernel<<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
              }
            }        
            else {
              index_t dims = std::accumulate(std::begin(sizes) + 1, std::end(sizes), 1, std::multiplies<index_t>());
              detail::matxOpTDKernel<<<blocks, threads, 0, stream_>>>(op, sizes, dims);
            } 
          }
#else
          #error "Cannot execute device function from host compiler"
#endif    
        }

    private:
      cudaStream_t stream_;
  };

};
