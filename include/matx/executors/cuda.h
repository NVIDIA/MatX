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
#include "matx/core/get_grid_dims.h"
#include "matx/executors/kernel.h"
#include "matx/executors/cuda_executor_common.h"
#include <cuda/std/array>
#include <utility>
#include <vector>

namespace matx
{

  /**
   * @brief Executes operators on a CUDA-enabled device (non-JIT path)
   * 
   * Optionally takes a stream for asynchronous execution.
   * This executor uses pre-compiled kernels without JIT compilation.
   * For JIT-based execution, use CUDAJit executor.
   * 
   */
  class cudaExecutor : public detail::CudaExecutorBase {
    public:
      /**
       * @brief Construct a new cudaExecutor with a stream
       * 
       * @param stream CUDA stream
       * @param profiling Whether to enable profiling
       */
      cudaExecutor(cudaStream_t stream, bool profiling = false) 
        : detail::CudaExecutorBase(stream, profiling) {}

      cudaExecutor(int stream, bool profiling = false) 
        : detail::CudaExecutorBase(stream, profiling) {}

      /**
       * @brief Construct a new cudaExecutor object using the default stream
       * 
       */
      cudaExecutor() : detail::CudaExecutorBase() {}

      
      /**
       * Execute an operator on a device (non-JIT path)
       * 
       * @tparam Op Operator type
       * @param op value
       **/
      template <typename Op>
        void Exec(const Op &op) const {
#ifdef __CUDACC__      
          dim3 threads = 1;
          dim3 blocks = 1;  

          // Parameters passed by value in CUDA are limited to CUDA_MAX_VAL_PARAM. If the user exceeds this, we 
          // need to error out and have them break up the statement
          if ((sizeof(op) + sizeof(index_t) * Op::Rank()) > detail::CUDA_MAX_VAL_PARAM) {
            MATX_THROW(matxInvalidParameter, 
                "Parameter buffer to device is limited to " + std::to_string(detail::CUDA_MAX_VAL_PARAM) + "B. "
                "Please break up your operator statement into multiple executions to limit the size of the parameters");
          }
          cuda::std::array<index_t, Op::Rank()> sizes;
          for (int i = 0; i < Op::Rank(); i++) {
            sizes[i] = op.Size(i);
          }   

          if constexpr (Op::Rank() <= 4) {
            // Create kernel provider for non-JIT
            auto kernel_provider = [&](detail::ElementsPerThread ept) {
              dim3 local_blocks = 1;
              dim3 local_threads = 1;
              bool stride = detail::get_grid_dims<Op::Rank()>(local_blocks, local_threads, sizes, static_cast<int>(ept), 256);
              
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
            auto [best_ept, shm_size, block_size, groups_per_block] = detail::find_best_launch_params(op, kernel_provider, 256, false);

            // Helper lambda to handle kernel dispatch. This is templated on the EPT
            // type since that's what the kernels are templated on.
            auto dispatch_kernel = [&]<detail::ElementsPerThread EPT>(auto&& kernel_handler) {
              int max_tpb = 256;

              bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(EPT), max_tpb);

              using CapType = detail::CapabilityParams<EPT, false>;
              
              if constexpr (Op::Rank() == 0) {
                kernel_handler([&]() {
                  detail::matxOpT0Kernel<CapType><<<blocks, threads, 0, stream_>>>(op);
                });
              }
              else if constexpr (Op::Rank() == 1) {
                kernel_handler([&]() {
                  detail::matxOpT1Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
                });
              }
              else if constexpr (Op::Rank() == 2) {
                if (stride) {
                  kernel_handler([&]() {
                    detail::matxOpT2StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                  });
                } else {
                  kernel_handler([&]() {
                    detail::matxOpT2Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                  });
                }
              }
              else if constexpr (Op::Rank() == 3) {
                if (stride) {
                  kernel_handler([&]() {
                    detail::matxOpT3StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                  });
                } else {
                  kernel_handler([&]() {
                    detail::matxOpT3Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                  });
                }
              }
              else if constexpr (Op::Rank() == 4) {
                if (stride) {
                  kernel_handler([&]() {
                    detail::matxOpT4StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                  });
                } else {
                  kernel_handler([&]() {
                    detail::matxOpT4Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                  });
                }
              }
            };

            // Helper lambda to launch kernel
            auto launch_kernel = [&]<detail::ElementsPerThread EPT>() {
              dispatch_kernel.template operator()<EPT>([&](auto launch_func) {
                launch_func();
              });
            };

            // Launch the correct kernel based on the best EPT found
            switch (best_ept) {
              case detail::ElementsPerThread::THIRTY_TWO:
                launch_kernel.template operator()<detail::ElementsPerThread::THIRTY_TWO>();
                break;
              case detail::ElementsPerThread::SIXTEEN:
                launch_kernel.template operator()<detail::ElementsPerThread::SIXTEEN>();
                break;
              case detail::ElementsPerThread::EIGHT:
                launch_kernel.template operator()<detail::ElementsPerThread::EIGHT>();
                break;
              case detail::ElementsPerThread::FOUR:
                launch_kernel.template operator()<detail::ElementsPerThread::FOUR>();
                break;
              case detail::ElementsPerThread::TWO:
                launch_kernel.template operator()<detail::ElementsPerThread::TWO>();
                break;
              case detail::ElementsPerThread::ONE:
                launch_kernel.template operator()<detail::ElementsPerThread::ONE>();
                break;
              default:
                MATX_THROW(matxInvalidParameter, "No kernel found for this operator");
            }
          }
          else {
            auto ept_type = detail::EPTQueryInput{false};
            const auto ept_bounds = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, ept_type);              
            bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(ept_bounds[1]), 1024);   
            index_t dims = cuda::std::accumulate(cuda::std::begin(sizes) + 1, cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
            detail::matxOpTDKernel<<<blocks, threads, 0, stream_>>>(op, sizes, dims);
          }            
#else
          MATX_ASSERT_STR(false, matxInvalidParameter, "Cannot call device executor using host compiler");
#endif    
        }
  };

  using CUDAExecutor = cudaExecutor; // Alias to make it consistent with host mode
};
