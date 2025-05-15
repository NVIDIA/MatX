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

#include "matx/core/capabilities.h"
#include "matx/core/defines.h"
#include "matx/executors/host.h"
#include "matx/core/capabilities.h"
#include "matx/core/nvrtc_helper.h"
#include "matx/core/get_grid_dims.h"
#include "matx/executors/kernel.h"
#include "matx/executors/cuda_executor_common.h"
#include <cuda/std/array>
#include <utility>
#include <vector>

namespace matx
{

  /**
   * @brief Executes operators on a CUDA-enabled device using JIT compilation
   * 
   * This executor uses NVRTC (NVIDIA Runtime Compilation) to JIT-compile kernels
   * at runtime for optimal performance. This is only available when MATX_EN_JIT
   * is defined.
   * 
   * Optionally takes a stream for asynchronous execution.
   * 
   */
  class CUDAJITExecutor : public detail::CudaExecutorBase {
    public:
      using jit_cuda_executor = bool;
      /**
       * @brief Construct a new CUDAJITExecutor executor with a stream
       * 
       * @param stream CUDA stream
       * @param profiling Whether to enable profiling
       */
      CUDAJITExecutor(cudaStream_t stream, bool profiling = false) 
        : detail::CudaExecutorBase(stream, profiling) {}

      CUDAJITExecutor(int stream, bool profiling = false) 
        : detail::CudaExecutorBase(stream, profiling) {}

      /**
       * @brief Construct a new CUDAJITExecutor executor using the default stream
       * 
       */
      CUDAJITExecutor() : detail::CudaExecutorBase() {}

      
      /**
       * Execute an operator on a device using JIT compilation
       * 
       * @tparam Op Operator type
       * @param op value
       **/
      template <typename Op>
        void Exec(const Op &op) const {
#ifdef MATX_EN_JIT
#ifdef __CUDACC__      
          dim3 threads = 1;
          dim3 blocks = 1;  

          // Parameters passed by value in CUDA are limited to CUDA_MAX_VAL_PARAM. If the user exceeds this, we 
          // need to error out and have them break up the statement
          MATX_STATIC_ASSERT((sizeof(op) + sizeof(index_t) * Op::Rank()) <= detail::CUDA_MAX_VAL_PARAM, 
              "Parameter buffer to device is limited to " + std::to_string(detail::CUDA_MAX_VAL_PARAM) + "B. "
              "Please break up your operator statement into multiple executions to limit the size of the parameters");

          cuda::std::array<index_t, Op::Rank()> sizes;
          for (int i = 0; i < Op::Rank(); i++) {
            sizes[i] = op.Size(i);
          }   

          if constexpr (Op::Rank() <= 4) {
            // Check if operator supports JIT
            // Force rank 2 or lower to account for weirdness when we have multiple groups per block
            bool use_jit = detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op) && Op::Rank() <= 2;
            
            if (!use_jit) {
              MATX_THROW(matxInvalidParameter, "Operator does not support JIT compilation. Use cudaExecutor instead.");
            }
            
            auto ept_type = detail::EPTQueryInput{true};
            const auto jit_ept_bounds = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, ept_type); 
            if (jit_ept_bounds[0] == detail::ElementsPerThread::INVALID) {
              MATX_THROW(matxInvalidParameter, "Operator does not support JIT compilation. Use cudaExecutor instead.");
            }
            
            // Create kernel provider for JIT
            auto kernel_provider = [&](detail::ElementsPerThread ept) {
              bool stride = detail::get_grid_dims_jit<Op::Rank()>(blocks, threads, sizes, static_cast<int>(ept), 1024, true);
              
              // Return appropriate kernel function pointer based on EPT, rank, and stride
              switch (ept) {
                case detail::ElementsPerThread::THIRTY_TWO:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>, Op>;
                  }
                  break;
                case detail::ElementsPerThread::SIXTEEN:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op>;
                  } else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>, Op>;
                  }
                  break;
                case detail::ElementsPerThread::EIGHT:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op>;
                  } else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>, Op>;
                  }
                  break;
                case detail::ElementsPerThread::FOUR:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op>;
                  }else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>, Op>;
                  }
                  break;
                case detail::ElementsPerThread::TWO:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::TWO, false>, Op>;
                  }
                  break;
                case detail::ElementsPerThread::ONE:
                  if constexpr (Op::Rank() == 0) {
                    return (const void*)detail::matxOpT0Kernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op>;
                  } else if constexpr (Op::Rank() == 1) {
                    return (const void*)detail::matxOpT1Kernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op>;
                  } else if constexpr (Op::Rank() == 2) {
                    return stride ? (const void*)detail::matxOpT2StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op> 
                                  : (const void*)detail::matxOpT2Kernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op>;
                  } else if constexpr (Op::Rank() == 3) {
                    return stride ? (const void*)detail::matxOpT3StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op> 
                                  : (const void*)detail::matxOpT3Kernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op>;
                  } else if constexpr (Op::Rank() == 4) {
                    return stride ? (const void*)detail::matxOpT4StrideKernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op> 
                                  : (const void*)detail::matxOpT4Kernel<detail::CapabilityParams<detail::ElementsPerThread::ONE, false>, Op>;
                  }
                  break;
                default:
                  return (const void*)nullptr;
              }
              return (const void*)nullptr;
            };

            // Find the best launch parameters
            auto [best_ept, shm_size, block_size, groups_per_block] = detail::find_best_launch_params(op, kernel_provider, 0, true);
                    
            bool stride = detail::get_grid_dims_jit<Op::Rank()>(blocks, threads, sizes, static_cast<int>(best_ept), groups_per_block, block_size, true);            
            //printf("shm_size %d stride %d  best_ept %d blocks %d %d %d\n", shm_size, stride, static_cast<int>(best_ept), blocks.x, threads.x, threads.y);
            const int osize = op.Rank() == 0 ? 1 : static_cast<int>(op.Size(op.Rank() - 1));
            detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, shm_size, osize);
          }
          else {
            MATX_THROW(matxInvalidParameter, "JIT compilation only supports operators with Rank <= 4");
          }            
#else
          MATX_ASSERT_STR(false, matxInvalidParameter, "Cannot call device executor using host compiler");
#endif    
#else
          MATX_THROW(matxInvalidParameter, "JIT compilation is not enabled. Define MATX_EN_JIT to use CUDAJit executor.");
#endif
        }
  };

}; // namespace matx
