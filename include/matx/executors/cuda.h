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
          printf("min/max ept %d %d\n", static_cast<int>(max_ept[0]), static_cast<int>(max_ept[1]));

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

#ifdef MATX_EN_JIT
            const bool use_jit = detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op) && Op::Rank() <= 4;
              printf("use_jit %d\n", use_jit);                  

            if (!use_jit)
#endif
            {
              
              if constexpr (Op::Rank() <= 4) {                
                //auto ept_tpb_options = get_max_threads(attr.numRegs, 1, attr.maxThreadsPerBlock);

                // Helper lambda to launch kernel. This is templated on the EPT
                // type since that's what the kernels are templated on.
                auto launch_kernel = [&]<detail::ElementsPerThread EPT>() {
                  int max_tpb = 1024;
                  // for(const auto &p : ept_tpb_options) {
                  //   if (p.first == static_cast<int>(EPT)) {
                  //     if (p.second > 0) {
                  //       max_tpb = p.second;
                  //     }
                  //     break;
                  //   }
                  // }

                  bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(EPT), max_tpb);
                  auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
                  printf("block_dim %lld %lld %lld\n", block_dim[0], block_dim[1], block_dim[2]);

                  using CapType = detail::CapabilityParams<EPT, false>;
                  cudaFuncAttributes attr;
                  cudaFuncGetAttributes(&attr, (const void*)detail::matxOpT1Kernel<CapType, Op>);
                  printf("numRegs %d maxThreadsPerBlock %d\n", attr.numRegs, attr.maxThreadsPerBlock);

                  if constexpr (Op::Rank() == 1) {
                    detail::matxOpT1Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
                  }
                  else if constexpr (Op::Rank() == 2) {
                    if (stride) {
                      detail::matxOpT2StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                    } else {
                      detail::matxOpT2Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                    }
                  }
                  else if constexpr (Op::Rank() == 3) {
                    if (stride) {
                      detail::matxOpT3StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                    } else {

                      printf("%lld %lld %lld blocks %d %d %d threads %d %d %d\n", sizes[0], sizes[1], sizes[2], blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
                      detail::matxOpT3Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                    }
                  }
                  else if constexpr (Op::Rank() == 4) {
                    if (stride) {
                      detail::matxOpT4StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                    } else {
                      detail::matxOpT4Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                    }
                  }
                };

                // Launch the correct kernel based on the max EPT
                switch (max_ept[1]) {
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
                bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(max_ept[1]), 1024);   
                index_t dims = cuda::std::accumulate(cuda::std::begin(sizes) + 1, cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
                detail::matxOpTDKernel<<<blocks, threads, 0, stream_>>>(op, sizes, dims);
              }
            }
#ifdef MATX_EN_JIT
            else {
              auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);              
              bool stride    = detail::get_grid_dims_jit<Op::Rank()>(blocks, threads, sizes, static_cast<int>(max_ept[1]), static_cast<int>(block_dim[2]), true);            
              int shm_size   = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op, detail::ShmQueryInput{static_cast<detail::ElementsPerThread>(max_ept[1])});
              printf("shm_size %d stride %d block_dim %lld %lld %lld\n", shm_size, stride, block_dim[0], block_dim[1], block_dim[2]);
              detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, static_cast<detail::ElementsPerThread>(max_ept[1]), stride, shm_size);
            }
#endif            
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
