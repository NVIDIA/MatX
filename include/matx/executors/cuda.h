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
#ifndef __CUDACC_RTC__
#pragma once

#include "matx/core/capabilities.h"
#include "matx/core/defines.h"
#include "matx/executors/host.h"
#include "matx/executors/kernel.h"
#include "matx/core/capabilities.h"
#include "matx/core/nvrtc_helper.h"
#include "matx/core/get_grid_dims.h"
#include <cuda/std/array>
#include <utility>
#include <vector>

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
      using cuda_executor = bool;  // signal this is a GPU executor
      using matx_executor = bool; ///< Type trait indicating this is an executor
      /**
       * @brief Construct a new cudaExecutor with a stream
       * 
       * @param stream CUDA stream
       * @param profiling Whether to enable profiling
       */
      cudaExecutor(cudaStream_t stream, bool profiling = false) : stream_(stream), profiling_(profiling) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      cudaExecutor(int stream, bool profiling = false) : stream_(reinterpret_cast<cudaStream_t>(stream)), profiling_(profiling) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      /**
       * @brief Construct a new cudaExecutor object using the default stream
       * 
       */
      cudaExecutor() : stream_(0), profiling_(false) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      ~cudaExecutor() {
        if (profiling_) {
          cudaEventDestroy(start_);
          cudaEventDestroy(stop_);
        }
      }

      /**
       * @brief Returns stream associated with executor
      */
      auto getStream() const { return stream_; }

      /**
       * @brief Synchronize the cuda executor's stream
       * 
       */
      void sync() { cudaStreamSynchronize(stream_); }

      /**
       * @brief Start a timer for profiling workload
       */
      void start_timer() { cudaEventRecord(start_, stream_); }

      /**
       * @brief Stop a timer for profiling workload
       */      
      void stop_timer() { cudaEventRecord(stop_, stream_); }

      /**
       * @brief Get the time in milliseconds between start_timer and stop_timer. 
       * This will block until the event is synchronized
       */
      float get_time_ms() {
        MATX_ASSERT_STR(profiling_, matxInvalidParameter, "Profiling not enabled when using get_time_ms()");
        float time;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&time, start_, stop_);
        return time;
      }

      /** 
       * With high EPT values the register pressure can increase, so we need to find a good value for enough occupancy. 
       * Target a minimum of 2 blocks per SM from register pressure.
       */
       template <typename T>
      [[nodiscard]] static __MATX_INLINE__ auto get_max_threads(int num_regs, int min_ept, int max_ept, int arch_max_tpb) {
        cuda::std::array<cuda::std::pair<int, int>, 6> ept_tpb_pairs;
        // int ept_tpb_idx = 0;
        // constexpr int min_occupancy = 2;

        // for (int ept = ept_start; ept <= 32; ept <<= 1) {
        //   int reg_estimate_per_thread = num_regs * ept;
        //   int current_threads = arch_max_tpb;

        //   // Find the highest tpb that allows min_occupancy
        //   while(current_threads > 32) { // 32 is min warp size
        //     if (reg_estimate_per_thread * current_threads * min_occupancy < arch_max_tpb * 256) { // TODO: Get max regs per block
        //       break;
        //     }
        //     current_threads >>= 1;
        //   }
        //   ept_tpb_pairs[ept_tpb_idx++] = {ept, current_threads};
        // }

        return ept_tpb_pairs;
      }

      /**
       * Find the best launch parameters by testing EPT values for optimal occupancy
       * 
       * @tparam Op Operator type
       * @tparam KernelProvider Function type that provides kernel function pointer for a given EPT
       * @param op Operator instance
       * @param max_ept Array containing min and max EPT values
       * @param kernel_provider Function that returns kernel function pointer for given EPT
       * @param use_jit Whether this is for JIT compilation (affects thread constraint logic)
       * @return Pair containing the best EPT and corresponding shared memory size
       */
      template <typename Op, typename KernelProvider>
      std::pair<detail::ElementsPerThread, int> find_best_launch_params(const Op &op, const cuda::std::array<detail::ElementsPerThread, 2>& max_ept, KernelProvider kernel_provider, bool use_jit = false) const {
        // Get device properties for constraints
        constexpr int min_occupancy = 2;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int max_dynamic_shm = static_cast<int>(prop.sharedMemPerBlock);
        int max_threads_per_block = static_cast<int>(prop.maxThreadsPerBlock);
        int regs_per_multiprocessor = static_cast<int>(prop.regsPerMultiprocessor);
        [[maybe_unused]]int block_size;
        
        // Start with maximum EPT and work down
        auto current_ept = max_ept[1];
        auto min_ept = max_ept[0];
        
        while (current_ept >= min_ept) {
          int shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op, detail::ShmQueryInput{current_ept});
          
          // Get kernel function pointer for this EPT and check register usage (if available)
          auto kernel_func = kernel_provider(current_ept);
          bool register_viable = true;  // Default to viable for JIT
          int num_regs = 0;  // Default for JIT or when kernel_func is null
          
          if (kernel_func != nullptr) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)kernel_func);
            num_regs = attr.numRegs;
            
            // Determine block size for register calculation
            block_size = max_threads_per_block;
            if (use_jit) {
              auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
              block_size = static_cast<int>(block_dim[2]);
            }
            
            // Check register pressure constraint: numRegs * block_size * 2 < regsPerMultiprocessor
            register_viable = (attr.numRegs * block_size * min_occupancy) < regs_per_multiprocessor;
          }
          
          // Check if launch is viable (2 blocks can be resident on SM)
          
          // Check dynamic shared memory constraint
          bool shm_viable = (shm_size * 2) < max_dynamic_shm;
          
          if (shm_viable && register_viable) {
            // printf("Selected EPT %d: registers %d, shm_size %d\n", 
            //        static_cast<int>(current_ept), num_regs, shm_size);
            return {current_ept, shm_size};
          }
          
          // printf("EPT %d failed constraints: shm_viable %d (%d of %d), register_viable %d (regs=%d) block size %d\n",
          //        static_cast<int>(current_ept), shm_viable, shm_size, max_dynamic_shm, register_viable, num_regs, block_size);
          
          // Cut EPT in half
          if (current_ept == detail::ElementsPerThread::ONE) break;
          
          switch (current_ept) {
            case detail::ElementsPerThread::THIRTY_TWO:
              current_ept = detail::ElementsPerThread::SIXTEEN;
              break;
            case detail::ElementsPerThread::SIXTEEN:
              current_ept = detail::ElementsPerThread::EIGHT;
              break;
            case detail::ElementsPerThread::EIGHT:
              current_ept = detail::ElementsPerThread::FOUR;
              break;
            case detail::ElementsPerThread::FOUR:
              current_ept = detail::ElementsPerThread::TWO;
              break;
            case detail::ElementsPerThread::TWO:
              current_ept = detail::ElementsPerThread::ONE;
              break;
            default:
              current_ept = detail::ElementsPerThread::ONE;
          }
        }
        
        // Fallback to minimum EPT
        int shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op, detail::ShmQueryInput{min_ept});
        //printf("Fallback to minimum EPT %d with shm_size %d\n", static_cast<int>(min_ept), shm_size);
        return {min_ept, shm_size};
      }
      
      /**
       * Execute an operator on a device
       * 
       * @tparam Op Operator type
       * @param op value
       **/
      template <typename Op>
        void Exec(const Op &op) const {
#ifdef __CUDACC__      
          dim3 threads, blocks;  

          // Parameters passed by value in CUDA are limited to CUDA_MAX_VAL_PARAM. If the user exceeds this, we 
          // need to error out and have them break up the statement
          MATX_STATIC_ASSERT((sizeof(op) + sizeof(index_t) * Op::Rank()) <= CUDA_MAX_VAL_PARAM, 
              "Parameter buffer to device is limited to " + std::to_string(CUDA_MAX_VAL_PARAM) + "B. "
              "Please break up your operator statement into multiple executions to limit the size of the parameters");

          const auto max_ept = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op);     
          //printf("min/max ept %d %d\n", static_cast<int>(max_ept[0]), static_cast<int>(max_ept[1]));

          if constexpr (Op::Rank() == 0) {
            threads = 1;
            blocks = 1;
            detail::matxOpT0Kernel<<<blocks, threads, 0, stream_>>>(op);
          }
          else {
            cuda::std::array<index_t, Op::Rank()> sizes;
            for (int i = 0; i < Op::Rank(); i++) {
              sizes[i] = op.Size(i);
            }   

            if constexpr (Op::Rank() <= 4) {
              // Create shared kernel provider generator function (accessible to both JIT and non-JIT)
              auto create_kernel_provider = [&](bool is_jit) {
                return [&, is_jit](detail::ElementsPerThread ept) {
                  // Determine if we'll use stride kernels
                  bool stride;
                  if (is_jit) {
                    stride = detail::get_grid_dims_jit<Op::Rank()>(blocks, threads, sizes, static_cast<int>(ept), 1024, true);
                  } else {
                    stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(ept), 1024);
                  }
                  
                  // Return appropriate kernel function pointer based on EPT, rank, and stride
                  switch (ept) {
                    case detail::ElementsPerThread::THIRTY_TWO:
                      if constexpr (Op::Rank() == 1) {
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
                      if constexpr (Op::Rank() == 1) {
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
                      if constexpr (Op::Rank() == 1) {
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
                      if constexpr (Op::Rank() == 1) {
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
                      if constexpr (Op::Rank() == 1) {
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
                      if constexpr (Op::Rank() == 1) {
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
              };

#ifdef MATX_EN_JIT
              const bool use_jit = detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op) && Op::Rank() <= 4;               

              if (!use_jit)
#endif
              {
                // Create kernel provider for non-JIT
                auto kernel_provider = create_kernel_provider(false);
                
                // Find the best launch parameters
                auto [best_ept, shm_size] = find_best_launch_params(op, max_ept, kernel_provider, false);
                //printf("Non-JIT best_ept %d shm_size %d\n", static_cast<int>(best_ept), shm_size);

                // Helper lambda to handle kernel dispatch. This is templated on the EPT
                // type since that's what the kernels are templated on.
                auto dispatch_kernel = [&]<detail::ElementsPerThread EPT>(auto&& kernel_handler) {
                  int max_tpb = 1024;

                  bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(EPT), max_tpb);
                  auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
                  //printf("block_dim %lld %lld %lld\n", block_dim[0], block_dim[1], block_dim[2]);

                  using CapType = detail::CapabilityParams<EPT, false>;
                  
                  if constexpr (Op::Rank() == 1) {
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
#ifdef MATX_EN_JIT
              else {
                // Create kernel provider for JIT using shared generator
                auto kernel_provider = create_kernel_provider(true);
                
                // Find the best launch parameters
                auto [best_ept, shm_size] = find_best_launch_params(op, max_ept, kernel_provider, true);
                
                auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);              
                bool stride    = detail::get_grid_dims_jit<Op::Rank()>(blocks, threads, sizes, static_cast<int>(best_ept), static_cast<int>(block_dim[2]), true);            
                //printf("shm_size %d stride %d block_dim %lld %lld %lld best_ept %d\n", shm_size, stride, block_dim[0], block_dim[1], block_dim[2], static_cast<int>(best_ept));
                detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, shm_size);
              }
#endif
            }
            else {
              bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(max_ept[1]), 1024);   
              index_t dims = cuda::std::accumulate(cuda::std::begin(sizes) + 1, cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
              detail::matxOpTDKernel<<<blocks, threads, 0, stream_>>>(op, sizes, dims);
            }            
          }
#else
          MATX_ASSERT_STR(false, matxInvalidParameter, "Cannot call device executor using host compiler");
#endif    
        }

    private:
      cudaStream_t stream_;
      bool profiling_;
      cudaEvent_t start_;
      cudaEvent_t stop_;
  };

  using CUDAExecutor = cudaExecutor; // Alias to make it consistent with host mode
};
#endif
