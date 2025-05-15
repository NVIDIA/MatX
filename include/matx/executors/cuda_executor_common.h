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
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&max_dynamic_shm, cudaDevAttrMaxSharedMemoryPerBlock , 0));
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0));
    MATX_CUDA_CHECK(cudaDeviceGetAttribute(&regs_per_multiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor, 0));

    const auto jit_query_in = detail::EPTQueryInput{use_jit};
    auto ept_bounds = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, jit_query_in);  

    // If we don't need async loads we don't want ILP to be higher than an individual vector load
    bool async_loads_requested = detail::get_operator_capability<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(op);
    if (!async_loads_requested) {
      int max_vec_load = detail::get_operator_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(op);
      ept_bounds[1] = static_cast<detail::ElementsPerThread>(cuda::std::min(static_cast<int>(ept_bounds[1]), max_vec_load));
    }

    // Start with maximum EPT and work down
    auto current_ept = ept_bounds[1];
    auto min_ept = ept_bounds[0];
    
    while (current_ept >= min_ept) {
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
        groups_per_block = group_range[0];            
        const auto set_groups_per_block_query = detail::SetGroupsPerBlockQueryInput{groups_per_block};
        const auto set_groups_per_block = detail::get_operator_capability<detail::OperatorCapability::SET_GROUPS_PER_BLOCK>(op, set_groups_per_block_query);            
        block_size = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
        shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
      }
      
      // Check register pressure constraint: numRegs * block_size * 2 < regsPerMultiprocessor
      register_viable = (attr.numRegs * block_size * min_occupancy) < regs_per_multiprocessor;
      
      // Check dynamic shared memory constraint
      bool shm_viable = (shm_size * 2) < max_dynamic_shm;
      
      if (shm_viable && register_viable) {
        // printf("Selected EPT %d: jits %d registers %d, shm_size %d, block_size %d, groups_per_block %d\n", 
        //        static_cast<int>(current_ept), use_jit, attr.numRegs, shm_size, block_size, groups_per_block);
        return cuda::std::make_tuple(current_ept, shm_size, block_size, groups_per_block);
      }
      else {
        // printf("EPT %d failed constraints: shm_viable %d (%d of %d), register_viable %d (regs=%d) block size %d, groups_per_block %d\n",
        //        static_cast<int>(current_ept), shm_viable, shm_size, max_dynamic_shm, register_viable, attr.numRegs, block_size, groups_per_block);
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

