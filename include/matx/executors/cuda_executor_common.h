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

#include <cuda_runtime.h>
#include "matx/core/capabilities.h"
#include "matx/core/defines.h"
#include "matx/core/get_grid_dims.h"
#include "matx/executors/kernel.h"
#include "matx/core/log.h"
#include <cuda/std/array>

namespace matx
{
namespace detail
{
  /**
   * @brief Base class for CUDA executors with common functionality
   * 
   * This class provides stream management, profiling, and synchronization
   * functionality that is shared between JIT and non-JIT CUDA executors.
   */
  class CudaExecutorBase {
    public:
      using cuda_executor = bool;  // signal this is a GPU executor
      using matx_executor = bool; ///< Type trait indicating this is an executor

      /**
       * @brief Construct a new CudaExecutorBase with a stream
       * 
       * @param stream CUDA stream
       * @param profiling Whether to enable profiling
       */
      CudaExecutorBase(cudaStream_t stream, bool profiling = false) : stream_(stream), profiling_(profiling) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      CudaExecutorBase(int stream, bool profiling = false) : stream_(reinterpret_cast<cudaStream_t>(stream)), profiling_(profiling) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      /**
       * @brief Construct a new CudaExecutorBase object using the default stream
       * 
       */
      CudaExecutorBase() : stream_(0), profiling_(false) {
        if (profiling_) {
          MATX_CUDA_CHECK(cudaEventCreate(&start_));
          MATX_CUDA_CHECK(cudaEventCreate(&stop_));
        }
      }

      ~CudaExecutorBase() {
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
      void start_timer() { 
        if (profiling_) {
          cudaEventRecord(start_, stream_); 
        }
      }

      /**
       * @brief Stop a timer for profiling workload
       */      
      void stop_timer() { 
        if (profiling_) {
          cudaEventRecord(stop_, stream_); 
        }
      }

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

    protected:
      cudaStream_t stream_;
      bool profiling_;
      cudaEvent_t start_;
      cudaEvent_t stop_;
  };

  /**
   * @brief Create a kernel provider that returns appropriate kernel function pointers
   * 
   * This function creates a lambda that provides kernel function pointers based on EPT,
   * rank, and stride parameters. Used by both JIT and non-JIT CUDA executors.
   * 
   * @tparam Op Operator type
   * @param sizes Array of dimension sizes
   * @param is_jit Whether this is for JIT compilation
   * @param global_kernel Whether this is a global kernel (only for JIT)
   * @return Lambda function that returns kernel pointer for a given EPT
   */
  template <typename Op>
  auto create_kernel_provider(const cuda::std::array<index_t, Op::Rank()>& sizes, bool is_jit = false, bool global_kernel = false) {
    return [&, is_jit, global_kernel](ElementsPerThread ept) -> const void* {
      dim3 local_blocks = 1;
      dim3 local_threads = 1;
      [[maybe_unused]] bool stride;
      
      if (is_jit && !global_kernel) {
        stride = get_grid_dims_block<Op::Rank()>(local_blocks, local_threads, sizes, static_cast<int>(ept), 1, 1024, true);
      } else {
        stride = get_grid_dims<Op::Rank()>(local_blocks, local_threads, sizes, static_cast<int>(ept), is_jit ? 1024 : 256);
      }

#ifdef __CUDACC__        
      // Return appropriate kernel function pointer based on EPT, rank, and stride
      switch (ept) {
        case ElementsPerThread::THIRTY_TWO:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::THIRTY_TWO, false>, Op>;
          }
          break;
        case ElementsPerThread::SIXTEEN:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::SIXTEEN, false>, Op>;
          }
          break;
        case ElementsPerThread::EIGHT:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::EIGHT, false>, Op>;
          }
          break;
        case ElementsPerThread::FOUR:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::FOUR, false>, Op>;
          }
          break;
        case ElementsPerThread::TWO:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::TWO, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::TWO, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::TWO, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::TWO, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::TWO, false>, Op>;
          }
          break;
        case ElementsPerThread::ONE:
          if constexpr (Op::Rank() == 0) {
            return (const void*)matxOpT0Kernel<CapabilityParams<ElementsPerThread::ONE, false>, Op>;
          } else if constexpr (Op::Rank() == 1) {
            return (const void*)matxOpT1Kernel<CapabilityParams<ElementsPerThread::ONE, false>, Op>;
          } else if constexpr (Op::Rank() == 2) {
            return stride ? (const void*)matxOpT2StrideKernel<CapabilityParams<ElementsPerThread::ONE, false>, Op> 
                          : (const void*)matxOpT2Kernel<CapabilityParams<ElementsPerThread::ONE, false>, Op>;
          } else if constexpr (Op::Rank() == 3) {
            return stride ? (const void*)matxOpT3StrideKernel<CapabilityParams<ElementsPerThread::ONE, false>, Op> 
                          : (const void*)matxOpT3Kernel<CapabilityParams<ElementsPerThread::ONE, false>, Op>;
          } else if constexpr (Op::Rank() == 4) {
            return stride ? (const void*)matxOpT4StrideKernel<CapabilityParams<ElementsPerThread::ONE, false>, Op> 
                          : (const void*)matxOpT4Kernel<CapabilityParams<ElementsPerThread::ONE, false>, Op>;
          }
          break;
        default:
          return (const void*)nullptr;
      }
#endif      
      return (const void*)nullptr;
    };
  }

  /**
   * Find the best launch parameters by testing EPT values for optimal occupancy
   * 
   * @tparam Op Operator type
   * @tparam KernelProvider Function type that provides kernel function pointer for a given EPT
   * @param op Operator instance
   * @param kernel_provider Function that returns kernel function pointer for given EPT
   * @param block_size Block size to use for the kernel
   * @param use_jit Whether this is for JIT compilation (affects thread constraint logic)
   * @return Tuple containing the best EPT, shared memory size, block size, and groups per block
   */
  template <typename Op, typename KernelProvider>
  auto find_best_launch_params(const Op &op, KernelProvider kernel_provider, int block_size, bool use_jit = false) {
    // Get device properties for constraints
    constexpr int min_occupancy = 2;
    int groups_per_block = 1;        
    int max_dynamic_shm, max_threads_per_block, regs_per_multiprocessor;
    int dev;
    MATX_CUDA_CHECK(cudaGetDevice(&dev));
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&max_dynamic_shm, cudaDevAttrMaxSharedMemoryPerBlock , dev));
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, dev));
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&regs_per_multiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor, dev));

    const auto jit_query_in = detail::EPTQueryInput{use_jit};
    auto ept_bounds = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, jit_query_in);  

    // If we don't need async loads we don't want ILP to be higher than an individual vector load
    bool async_loads_requested = detail::get_operator_capability<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(op);
    if (!async_loads_requested) {
      int max_vec_load = detail::get_operator_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(op);
      ept_bounds[1] = static_cast<detail::ElementsPerThread>(cuda::std::min(static_cast<int>(ept_bounds[1]), max_vec_load));
      MATX_LOG_DEBUG("Async loads not needed. Max EPT for vector load: {}", max_vec_load);
    }

    // Start with maximum EPT and work down
    auto current_ept = ept_bounds[1];
    auto min_ept = ept_bounds[0];
    
    while (current_ept >= min_ept) {
      MATX_LOG_TRACE("Finding best launch params: current_ept {}, min_ept {}", static_cast<int>(current_ept), static_cast<int>(min_ept));
      int shm_size = 0; // Default to no shm since non-JIT doesn't use any
      // Get kernel function pointer for this EPT and check register usage (if available)
      auto kernel_func = kernel_provider(current_ept);
      bool register_viable = true;  // Default to viable for JIT

      cudaFuncAttributes attr;
      MATX_CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*)kernel_func));

      const auto set_ept_query = detail::SetEPTQueryInput{current_ept};
      const auto set_ept = detail::get_operator_capability<detail::OperatorCapability::SET_ELEMENTS_PER_THREAD>(op, set_ept_query);
      
      // Determine block size for register calculation
      if (use_jit) {
        const auto group_range = detail::get_operator_capability<detail::OperatorCapability::GROUPS_PER_BLOCK>(op);
        int min_groups_per_block = group_range[0];
        int max_groups_per_block = group_range[1];
        if (max_groups_per_block == 32) {
          max_groups_per_block = 1024;
        }

        int total_batches = 1;
        if constexpr (Op::Rank() > 0) {
          total_batches = static_cast<int>(TotalSize(op) / op.Size(op.Rank() - 1));
        }

        // Iterate through all possible groups_per_block values
        for (int current_groups_per_block = max_groups_per_block; current_groups_per_block >= min_groups_per_block; current_groups_per_block /= 2) {
          // If we don't have enough batches then skip this groups_per_block
          if (current_groups_per_block > total_batches) {
            continue;
          }

          MATX_LOG_DEBUG("Trying groups_per_block {} with {} batches", current_groups_per_block, total_batches);
          
          groups_per_block = current_groups_per_block;
          const auto set_groups_per_block_query = detail::SetGroupsPerBlockQueryInput{groups_per_block};
          const auto set_groups_per_block = detail::get_operator_capability<detail::OperatorCapability::SET_GROUPS_PER_BLOCK>(op, set_groups_per_block_query);            
          // Use the max block size for now
          block_size = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op)[1];
          shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
          
          // Check register pressure constraint
          register_viable = (attr.numRegs * block_size * min_occupancy) <= regs_per_multiprocessor;
          
          // Check dynamic shared memory constraint
          bool shm_viable = (shm_size * 2) < max_dynamic_shm;
          
          if (shm_viable && register_viable) {
            MATX_LOG_DEBUG("Selected EPT {}: jits {}, registers {}, shm_size {}, block_size {}, groups_per_block {}", 
                           static_cast<int>(current_ept), use_jit, attr.numRegs, shm_size, block_size, groups_per_block);
            return cuda::std::make_tuple(current_ept, shm_size, block_size, groups_per_block);
          }
          else {
            MATX_LOG_DEBUG("EPT {} with groups_per_block {} failed constraints: shm_viable {} ({} of {}), register_viable {} (regs={}) block size {}",
                           static_cast<int>(current_ept), groups_per_block, shm_viable, shm_size, max_dynamic_shm, register_viable, attr.numRegs, block_size);
          }
          
          // Break if we're at the minimum
          if (current_groups_per_block == min_groups_per_block) break;
        }
      }
      else {
        // Non-JIT path - check constraints without groups_per_block loop
        // Check register pressure constraint
        register_viable = (attr.numRegs * block_size * min_occupancy) <= regs_per_multiprocessor;
        
        // Check dynamic shared memory constraint
        bool shm_viable = (shm_size * 2) < max_dynamic_shm;
        
        if (shm_viable && register_viable) {
          MATX_LOG_DEBUG("Selected EPT {}: jits {}, registers {}, shm_size {}, block_size {}, groups_per_block {}", 
                         static_cast<int>(current_ept), use_jit, attr.numRegs, shm_size, block_size, groups_per_block);
          return cuda::std::make_tuple(current_ept, shm_size, block_size, groups_per_block);
        }
        else {
          MATX_LOG_DEBUG("EPT {} failed constraints: shm_viable {} ({} of {}), register_viable {} (regs={}) block size {}, groups_per_block {}",
                         static_cast<int>(current_ept), shm_viable, shm_size, max_dynamic_shm, register_viable, attr.numRegs, block_size, groups_per_block);
        }
      }

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
    const auto set_ept_query = detail::SetEPTQueryInput{min_ept};
    const auto set_ept = detail::get_operator_capability<detail::OperatorCapability::SET_ELEMENTS_PER_THREAD>(op, set_ept_query);
    int shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
    //printf("Fallback to minimum EPT %d with shm_size %d\n", static_cast<int>(min_ept), shm_size);
    return cuda::std::make_tuple(min_ept, shm_size, block_size, groups_per_block);
  }

} // namespace detail
} // namespace matx

