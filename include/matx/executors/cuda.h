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
#include "matx/core/capabilities.h"

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
            if constexpr (Op::Rank() == 1) {
              if (max_ept == detail::ElementsPerThread::ONE) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else if (max_ept == detail::ElementsPerThread::TWO) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else if (max_ept == detail::ElementsPerThread::FOUR) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                detail::matxOpT1Kernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0]);
              }
            }
            else if constexpr (Op::Rank() == 2) {
              if(stride) {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT2StrideKernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                }
              } else {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT2Kernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1]);
                }
              }
            }
            else if constexpr (Op::Rank() == 3) {
              if(stride) {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT3StrideKernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                }
              } else {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT3Kernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2]);
                }
              }
            }
            else if constexpr (Op::Rank() == 4) {
              if(stride) {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT4StrideKernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                }
              } else {
                if (max_ept == detail::ElementsPerThread::ONE) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::ONE><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::TWO) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::FOUR) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::FOUR><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::EIGHT) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::EIGHT><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::SIXTEEN) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::SIXTEEN><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                } else if (max_ept == detail::ElementsPerThread::THIRTY_TWO) {
                  detail::matxOpT4Kernel<detail::ElementsPerThread::THIRTY_TWO><<<blocks, threads, 0, stream_>>>(op, sizes[0], sizes[1], sizes[2], sizes[3]);
                }
              }
            }        
            else {
              index_t dims = std::accumulate(std::begin(sizes) + 1, std::end(sizes), 1, std::multiplies<index_t>());
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
