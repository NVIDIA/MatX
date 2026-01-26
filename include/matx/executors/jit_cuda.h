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
#include <unordered_map>
#include <mutex>

namespace matx
{
  namespace detail {
    /**
     * @brief Cached launch parameters for JIT kernels
     * 
     * This structure stores all computed launch parameters so we can skip expensive
     * computations when we've already compiled a kernel for this operator type.
     * 
     * IMPORTANT: For JIT compilation, tensor sizes are encoded in the operator type
     * string (from JIT_TYPE_QUERY). This means different sizes produce different cache
     * keys, so grid dimensions ARE safe to cache - they won't vary for the same type.
     * 
     * The cache works in conjunction with the nvrtc_compile_and_run kernel cache:
     * 1. First execution: Computes all launch params -> caches everything
     * 2. Subsequent executions: Uses cached params -> skips ALL computations
     * 
     * This avoids:
     * - find_best_launch_params: EPT selection, device queries, occupancy calculations
     * - get_grid_dims/get_grid_dims_block: Grid dimension calculations
     * - All CUDA device attribute queries
     * - Register pressure and shared memory constraint analysis
     * 
     * The cache is keyed by the operator type string from JIT_TYPE_QUERY which
     * includes both the operator structure AND tensor sizes.
     */
    struct JITLaunchParams {
      ElementsPerThread best_ept;       // Optimal elements per thread for this operator
      int shm_size;                      // Dynamic shared memory size in bytes
      int block_size;                    // Block dimension size
      int groups_per_block;              // Groups per block (for block-level kernels)
      bool stride;                       // Whether kernel uses grid-stride loops
      dim3 blocks;                       // Grid dimensions (x, y, z blocks)
      dim3 threads;                      // Block dimensions (x, y, z threads)
      int osize;                         // Output size (last dimension)
      bool global_kernel;                // Whether this is a global or block-level kernel
      bool pass_through_threads;         // Whether all threads must call operator() (bounds checking at tensor level)
    };

    // Global cache for JIT launch parameters, keyed by operator type string from JIT_TYPE_QUERY
    static std::unordered_map<std::string, JITLaunchParams> jit_launch_params_cache;
    static std::mutex jit_launch_params_mutex;
  }  // namespace detail

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
          if ((sizeof(op) + sizeof(index_t) * Op::Rank()) > detail::CUDA_MAX_VAL_PARAM) {
            MATX_THROW(matxInvalidParameter, 
                "Parameter buffer to device is limited to " + std::to_string(detail::CUDA_MAX_VAL_PARAM) + "B. "
                "Please break up your operator statement into multiple executions to limit the size of the parameters");
          }

          cuda::std::array<index_t, Op::Rank()> sizes;
          for (int i = 0; i < Op::Rank(); i++) {
            sizes[i] = op.Size(i);
          }   

          // Check if operator supports JIT
          // Force rank 2 or lower to account for weirdness when we have multiple groups per block
          bool use_jit = detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op);

          MATX_LOG_DEBUG("Using JIT: {}", use_jit);
          if (!use_jit) {
            MATX_THROW(matxInvalidParameter, "Operator does not support JIT compilation. Use cudaExecutor instead.");
          }
          
          auto ept_type = detail::EPTQueryInput{true};
          const auto jit_ept_bounds = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, ept_type); 
          if (jit_ept_bounds[0] == detail::ElementsPerThread::INVALID) {
            MATX_THROW(matxInvalidParameter, "Operator does not support JIT compilation. Use cudaExecutor instead.");
          }


          bool global_kernel = detail::get_operator_capability<detail::OperatorCapability::GLOBAL_KERNEL>(op);
          bool pass_through_threads = detail::get_operator_capability<detail::OperatorCapability::PASS_THROUGH_THREADS>(op);
          if (global_kernel) {
            MATX_LOG_DEBUG("Operator operates on a global level");
          } else if (pass_through_threads) {
            MATX_LOG_DEBUG("Operator uses pass-through threads (bounds checking at tensor level)");
          } else {
            MATX_LOG_DEBUG("Operator operates on a block level");
          }

          if constexpr (Op::Rank() <= 4) {
            // Get operator type string for cache lookup
            const auto kernel_op_type = detail::get_operator_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(op);
            
            // Check if we have cached launch parameters for this operator type
            detail::JITLaunchParams cached_params;
            bool has_cached_params = false;
            {
              std::lock_guard<std::mutex> lock(detail::jit_launch_params_mutex);
              auto it = detail::jit_launch_params_cache.find(kernel_op_type);
              if (it != detail::jit_launch_params_cache.end()) {
                cached_params = it->second;
                has_cached_params = true;
              }
            }

            detail::ElementsPerThread best_ept;
            int shm_size, block_size, groups_per_block;
            bool stride;
            
            if (has_cached_params) {
              // Use cached parameters - skip ALL expensive computations!
              MATX_LOG_DEBUG("Using cached launch parameters for operator type: {}", kernel_op_type);
              best_ept = cached_params.best_ept;
              shm_size = cached_params.shm_size;
              block_size = cached_params.block_size;
              groups_per_block = cached_params.groups_per_block;
              stride = cached_params.stride;
              blocks = cached_params.blocks;
              threads = cached_params.threads;
              
              MATX_LOG_DEBUG("Cached EPT {}, Shm size {}, Block size {}, Groups per block {}", 
                             static_cast<int>(best_ept), shm_size, block_size, groups_per_block);
            } else {
              // No cached parameters - compute them
              MATX_LOG_DEBUG("No cached parameters found, computing launch parameters for JIT");
              
              if (pass_through_threads) {
                // For pass-through operators (e.g., cuBLASDx), block dimensions are fixed by the operator
                auto block_dim_range = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
                block_size = block_dim_range[1];  // Use the max block size
                stride = detail::get_grid_dims_block_2d<Op::Rank()>(blocks, threads, sizes, block_size);
                
                // EPT is 1 for 2D block operators - the operator handles elements internally
                best_ept = detail::ElementsPerThread::ONE;
                shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
                groups_per_block = 1;
                
                MATX_LOG_DEBUG("Block2D: EPT {}, Shm size {}, Block size {}", 
                               static_cast<int>(best_ept), shm_size, block_size);
              } else {
                // Create kernel provider for JIT using consolidated function
                auto kernel_provider = detail::create_kernel_provider<Op>(sizes, true, global_kernel);

                // Find the best launch parameters
                auto result = detail::find_best_launch_params(op, kernel_provider, 0, true);
                best_ept = cuda::std::get<0>(result);
                shm_size = cuda::std::get<1>(result);
                block_size = cuda::std::get<2>(result);
                groups_per_block = cuda::std::get<3>(result);
                
                MATX_LOG_DEBUG("Best EPT {}, Shm size {}, Block size {}, Groups per block {}", 
                               static_cast<int>(best_ept), shm_size, block_size, groups_per_block);
                
                if (global_kernel) {
                  stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(best_ept), 256);
                } else {
                  stride = detail::get_grid_dims_block<Op::Rank()>(blocks, threads, sizes, static_cast<int>(best_ept), groups_per_block, block_size, true);
                }
              }
              
              // Cache ALL parameters for future use (sizes are encoded in type string)
              detail::JITLaunchParams params_to_cache;
              params_to_cache.best_ept = best_ept;
              params_to_cache.shm_size = shm_size;
              params_to_cache.block_size = block_size;
              params_to_cache.groups_per_block = groups_per_block;
              params_to_cache.stride = stride;
              params_to_cache.blocks = blocks;
              params_to_cache.threads = threads;
              params_to_cache.osize = op.Rank() == 0 ? 1 : static_cast<int>(op.Size(op.Rank() - 1));
              params_to_cache.global_kernel = global_kernel;
              params_to_cache.pass_through_threads = pass_through_threads;
              
              {
                std::lock_guard<std::mutex> lock(detail::jit_launch_params_mutex);
                detail::jit_launch_params_cache[kernel_op_type] = params_to_cache;
              }
            }

            MATX_LOG_DEBUG("Shm size {}, Stride {}, estimated EPT {}, blocks {}x{}x{} threads {}x{}x{}, pass_through_threads {}", 
                shm_size, stride, static_cast<int>(best_ept), blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, pass_through_threads);
            const int osize = op.Rank() == 0 ? 1 : static_cast<int>(op.Size(op.Rank() - 1));
            detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, shm_size, osize, global_kernel, pass_through_threads);
          }
          else {
            // ND kernel support for ranks > 4 (JIT path)
            // Get operator type string for cache lookup
            const auto kernel_op_type = detail::get_operator_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(op);
            
            // Check if we have cached launch parameters for this operator type
            detail::JITLaunchParams cached_params;
            bool has_cached_params = false;
            {
              std::lock_guard<std::mutex> lock(detail::jit_launch_params_mutex);
              auto it = detail::jit_launch_params_cache.find(kernel_op_type);
              if (it != detail::jit_launch_params_cache.end()) {
                cached_params = it->second;
                has_cached_params = true;
              }
            }
            
            detail::ElementsPerThread best_ept;
            bool stride;
            
            if (has_cached_params) {
              // Use cached parameters - skip ALL computations!
              MATX_LOG_DEBUG("Using cached launch parameters for ND kernel: {}", kernel_op_type);
              best_ept = cached_params.best_ept;
              stride = cached_params.stride;
              blocks = cached_params.blocks;
              threads = cached_params.threads;
            } else {
              // No cached parameters - compute them
              MATX_LOG_DEBUG("No cached parameters found, computing launch parameters for ND kernel");
              
              // Reuse the ept_type and jit_ept_bounds from above
              const auto ept_bounds = jit_ept_bounds;
              best_ept = ept_bounds[1];
              stride = detail::get_grid_dims<Op::Rank()>(blocks, threads, sizes, static_cast<int>(best_ept), 1024);   
              
              // Cache ALL parameters for future use (sizes are encoded in type string)
              detail::JITLaunchParams params_to_cache;
              params_to_cache.best_ept = best_ept;
              params_to_cache.shm_size = 0;
              params_to_cache.block_size = threads.x;
              params_to_cache.groups_per_block = 1;
              params_to_cache.stride = stride;
              params_to_cache.blocks = blocks;
              params_to_cache.threads = threads;
              params_to_cache.osize = op.Rank() == 0 ? 1 : static_cast<int>(op.Size(op.Rank() - 1));
              params_to_cache.global_kernel = true;
              params_to_cache.pass_through_threads = pass_through_threads;
              {
                std::lock_guard<std::mutex> lock(detail::jit_launch_params_mutex);
                detail::jit_launch_params_cache[kernel_op_type] = params_to_cache;
              }
            }
            
            MATX_LOG_DEBUG("Using ND kernel for rank > 4 with JIT and EPT {}", static_cast<int>(best_ept));            
            index_t dims = cuda::std::accumulate(cuda::std::begin(sizes) + 1, cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
            const int osize = op.Rank() == 0 ? 1 : static_cast<int>(op.Size(op.Rank() - 1));
            
            MATX_LOG_DEBUG("ND kernel: stride {}, blocks {}x{}x{} threads {}x{}x{}, dims {}", 
                stride, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, dims);
            
            // Use ND kernel through JIT compilation
            detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, 0, osize, true);
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
