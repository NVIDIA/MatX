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
#include "matx/core/nvrtc.h"
#include "matx/core/get_grid_dims.h"

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
       * Execute an operator on a device
       * 
       * @tparam Op Operator type
       * @param op value
       **/
      template <typename Op>
        void Exec(const Op &op) const {
#ifdef __CUDACC__      
          dim3 threads, blocks;  

          // Parameters passed by value in CUDA are limited to 4096B. If the user exceeds this, we 
          // need to error out and have them break up the statement
          MATX_STATIC_ASSERT((sizeof(op) + sizeof(index_t) * Op::Rank()) <= CUDA_MAX_VAL_PARAM, 
              "Parameter buffer to device is limited to 4096B. Please break up your operator statement into multiple executions to limit the size of the parameters");

          const auto max_ept = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op);     
          printf("max ept %d\n", static_cast<int>(max_ept));    

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

            bool stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(max_ept), 256);
            
            // Helper function to execute kernel with dual path (direct kernel vs JIT)
            auto execute_with_ept = [&](auto ept_tag) -> bool {
              constexpr auto EPT = decltype(ept_tag)::ept;
              using CapType      = decltype(ept_tag);
              if (max_ept == EPT) {
                if constexpr (Op::Rank() == 1) {
#ifdef MATX_EN_MATHDX
                  if constexpr (EPT == detail::ElementsPerThread::TWO) {
                    nvrtc_compile_and_run<CapType>(matx::detail::matxOpT1JITKernelStr, "output.cu", op, sizes[0], blocks, threads);
                  } else {
                    detail::matxOpT1Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
                  }
#else
                  detail::matxOpT1Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
#endif
                }
                else if constexpr (Op::Rank() == 2) {
                  if (stride) {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 2D stride kernels if needed
                    detail::matxOpT2StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
#else
                    detail::matxOpT2StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
#endif
                  } else {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 2D kernels if needed
                    detail::matxOpT2Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
#else
                    detail::matxOpT2Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
#endif
                  }
                }
                else if constexpr (Op::Rank() == 3) {
                  if (stride) {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 3D stride kernels if needed
                    detail::matxOpT3StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
#else
                    detail::matxOpT3StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
#endif
                  } else {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 3D kernels if needed
                    detail::matxOpT3Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
#else
                    detail::matxOpT3Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
#endif
                  }
                }
                else if constexpr (Op::Rank() == 4) {
                  if (stride) {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 4D stride kernels if needed
                    detail::matxOpT4StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
#else
                    detail::matxOpT4StrideKernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
#endif
                  } else {
#ifdef MATX_EN_MATHDX
                    // Add JIT path for 4D kernels if needed
                    detail::matxOpT4Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
#else
                    detail::matxOpT4Kernel<CapType><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
#endif
                  }
                }
                return true;
              }
              return false;
            };

            // Helper tags for template parameter deduction
            constexpr auto one_tag = detail::CapabilityParams<detail::ElementsPerThread::ONE, false>{};
            constexpr auto two_tag = detail::CapabilityParams<detail::ElementsPerThread::TWO, false>{};
            constexpr auto four_tag = detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>{};
            constexpr auto eight_tag = detail::CapabilityParams<detail::ElementsPerThread::EIGHT, false>{};
            constexpr auto sixteen_tag = detail::CapabilityParams<detail::ElementsPerThread::SIXTEEN, false>{};
            constexpr auto thirty_two_tag = detail::CapabilityParams<detail::ElementsPerThread::THIRTY_TWO, false>{};

            // Try each EPT value in order
            if (execute_with_ept(thirty_two_tag) ||
                execute_with_ept(sixteen_tag) ||
                execute_with_ept(eight_tag) ||
                execute_with_ept(four_tag) ||
                execute_with_ept(two_tag) ||
                execute_with_ept(one_tag)) {
              // Successfully executed
            }
            else if constexpr (Op::Rank() > 4) {
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
