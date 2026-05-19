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
#include <sstream>
#include <optional>

namespace matx
{
  namespace detail {
#ifdef MATX_EN_JIT
    /**
     * @brief Cached launch parameters for JIT kernels
     * 
     * This structure stores all computed launch parameters so we can skip expensive
     * computations when we've already compiled a kernel for this operator type.
     * 
     * IMPORTANT: For JIT compilation, tensor sizes are encoded in either the fixed-size
     * JIT_CACHE_KEY fingerprint or the fallback operator type string (from JIT_TYPE_QUERY).
     * This means different sizes produce different cache keys, so grid dimensions ARE
     * safe to cache - they won't vary for the same key.
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
     * Operators that support JIT_CACHE_KEY use that fixed-size key to avoid constructing
     * the full JIT_TYPE_QUERY string on warmed launches. Other operators use the
     * original string key.
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
      int pass_through_inner_rank;       // Cooperative trailing dimensions for pass-through kernels
      bool block_reduces_rank;           // Whether a block collective reduces an output rank dimension
    };

    // Global cache for JIT launch parameters, keyed by JIT_CACHE_KEY when available,
    // with the original JIT_TYPE_QUERY string as a fallback.
    struct JITLaunchParamsCacheKey {
      JITCacheKey op_key;
      int device_id = 0;
      std::string sm_arch;

      __MATX_INLINE__ __MATX_HOST__ bool operator==(const JITLaunchParamsCacheKey &rhs) const noexcept
      {
        return device_id == rhs.device_id && sm_arch == rhs.sm_arch && op_key == rhs.op_key;
      }
    };

    struct JITLaunchParamsCacheKeyHash {
      __MATX_INLINE__ __MATX_HOST__ std::size_t operator()(const JITLaunchParamsCacheKey &key) const noexcept
      {
        std::size_t h = JITCacheKeyHash{}(key.op_key);
        h ^= static_cast<std::size_t>(key.device_id) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(key.sm_arch) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
      }
    };

    inline std::unordered_map<std::string, JITLaunchParams> jit_launch_params_cache;
    inline std::unordered_map<JITLaunchParamsCacheKey, JITLaunchParams, JITLaunchParamsCacheKeyHash> jit_launch_params_cache_by_key;
    inline std::mutex jit_launch_params_mutex;

    static constexpr int JIT_LAUNCH_PARAMS_METADATA_VERSION = 6;

    inline int GetJITLaunchParamsDevice()
    {
      int device = 0;
      MATX_CUDA_CHECK(cudaGetDevice(&device));
      return device;
    }

    inline std::string GetJITLaunchParamsDevicePrefix()
    {
      return "device_" + std::to_string(GetJITLaunchParamsDevice()) + "_sm_" + resolve_nvrtc_cuda_arch() + "_";
    }

    inline std::string GetJITLaunchParamsCacheKey(const std::string &kernel_op_type)
    {
      return GetJITLaunchParamsDevicePrefix() + kernel_op_type;
    }

    inline JITLaunchParamsCacheKey GetJITLaunchParamsCacheKey(const JITCacheKey &jit_cache_key)
    {
      return JITLaunchParamsCacheKey{jit_cache_key, GetJITLaunchParamsDevice(), resolve_nvrtc_cuda_arch()};
    }

    inline std::string GetJITLaunchParamsMetadataFilename(const std::string &kernel_op_type)
    {
      return GetCache().TypeStringToFilename(GetJITLaunchParamsCacheKey(kernel_op_type)) + ".launch";
    }

    inline std::string GetJITLaunchParamsMetadataFilename(const JITCacheKey &jit_cache_key)
    {
      return GetJITLaunchParamsDevicePrefix() + JITCacheKeyToFilename(jit_cache_key, "JITLaunch") + ".launch";
    }

    inline std::string SerializeJITLaunchParams(const JITLaunchParams &params)
    {
      std::ostringstream os;
      os << JIT_LAUNCH_PARAMS_METADATA_VERSION << ' '
         << static_cast<int>(params.best_ept) << ' '
         << params.shm_size << ' '
         << params.block_size << ' '
         << params.groups_per_block << ' '
         << static_cast<int>(params.stride) << ' '
         << params.blocks.x << ' ' << params.blocks.y << ' ' << params.blocks.z << ' '
         << params.threads.x << ' ' << params.threads.y << ' ' << params.threads.z << ' '
         << params.osize << ' '
         << static_cast<int>(params.global_kernel) << ' '
         << static_cast<int>(params.pass_through_threads) << ' '
         << params.pass_through_inner_rank << ' '
         << static_cast<int>(params.block_reduces_rank);
      return os.str();
    }

    inline bool DeserializeJITLaunchParams(const std::string &metadata, JITLaunchParams &params)
    {
      if (metadata.empty()) {
        return false;
      }

      int version = 0;
      int ept = 0;
      int stride = 0;
      int global_kernel = 0;
      int pass_through_threads = 0;
      int pass_through_inner_rank = 2;
      int block_reduces_rank = 0;
      unsigned int blocks_x = 0;
      unsigned int blocks_y = 0;
      unsigned int blocks_z = 0;
      unsigned int threads_x = 0;
      unsigned int threads_y = 0;
      unsigned int threads_z = 0;

      std::istringstream is(metadata);
      if (!(is >> version
            >> ept
            >> params.shm_size
            >> params.block_size
            >> params.groups_per_block
            >> stride
            >> blocks_x >> blocks_y >> blocks_z
            >> threads_x >> threads_y >> threads_z
            >> params.osize
            >> global_kernel
            >> pass_through_threads)) {
        return false;
      }

      if (version >= 2 && !(is >> pass_through_inner_rank)) {
        return false;
      }
      if (version >= 5 && !(is >> block_reduces_rank)) {
        return false;
      }

      if (version != JIT_LAUNCH_PARAMS_METADATA_VERSION ||
          params.shm_size < 0 ||
          params.block_size <= 0 ||
          params.groups_per_block <= 0 ||
          threads_x == 0 || threads_y == 0 || threads_z == 0 ||
          pass_through_inner_rank < 1 || pass_through_inner_rank > 2) {
        return false;
      }

      params.best_ept = static_cast<ElementsPerThread>(ept);
      params.stride = stride != 0;
      params.blocks = dim3{blocks_x, blocks_y, blocks_z};
      params.threads = dim3{threads_x, threads_y, threads_z};
      params.global_kernel = global_kernel != 0;
      params.pass_through_threads = pass_through_threads != 0;
      params.pass_through_inner_rank = pass_through_inner_rank;
      params.block_reduces_rank = block_reduces_rank != 0;
      return params.best_ept != ElementsPerThread::INVALID;
    }

    inline bool LookupJITLaunchParams(const std::string &kernel_op_type, JITLaunchParams &params)
    {
      const auto cache_key = GetJITLaunchParamsCacheKey(kernel_op_type);
      {
        std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
        auto it = jit_launch_params_cache.find(cache_key);
        if (it != jit_launch_params_cache.end()) {
          params = it->second;
          return true;
        }
      }

      auto metadata = GetCache().GetLTOIRMetadata(GetJITLaunchParamsMetadataFilename(kernel_op_type));
      if (!DeserializeJITLaunchParams(metadata, params)) {
        return false;
      }

      std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
      jit_launch_params_cache[cache_key] = params;
      return true;
    }

    inline bool LookupJITLaunchParams(const JITCacheKey &jit_cache_key, JITLaunchParams &params)
    {
      if (!jit_cache_key.valid) {
        return false;
      }
      const auto cache_key = GetJITLaunchParamsCacheKey(jit_cache_key);

      {
        std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
        auto it = jit_launch_params_cache_by_key.find(cache_key);
        if (it != jit_launch_params_cache_by_key.end()) {
          params = it->second;
          return true;
        }
      }

      auto metadata = GetCache().GetLTOIRMetadata(GetJITLaunchParamsMetadataFilename(jit_cache_key));
      if (!DeserializeJITLaunchParams(metadata, params)) {
        return false;
      }

      std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
      jit_launch_params_cache_by_key[cache_key] = params;
      return true;
    }

    inline void StoreJITLaunchParams(const std::string &kernel_op_type, const JITLaunchParams &params)
    {
      const auto cache_key = GetJITLaunchParamsCacheKey(kernel_op_type);
      {
        std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
        jit_launch_params_cache[cache_key] = params;
      }

      GetCache().StoreLTOIRMetadata(GetJITLaunchParamsMetadataFilename(kernel_op_type),
                                    SerializeJITLaunchParams(params));
    }

    inline void StoreJITLaunchParams(const JITCacheKey &jit_cache_key, const JITLaunchParams &params)
    {
      if (!jit_cache_key.valid) {
        return;
      }
      const auto cache_key = GetJITLaunchParamsCacheKey(jit_cache_key);

      {
        std::lock_guard<std::mutex> lock(jit_launch_params_mutex);
        jit_launch_params_cache_by_key[cache_key] = params;
      }

      GetCache().StoreLTOIRMetadata(GetJITLaunchParamsMetadataFilename(jit_cache_key),
                                    SerializeJITLaunchParams(params));
    }
#endif
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
      /**
       * Execute an operator, dispatching on runtime rank for dynamic tensors.
       */
      template <typename Op>
        void Exec(const Op &op) const {
#ifdef MATX_EN_JIT
#ifdef __CUDACC__
          if constexpr (is_dynamic_rank_op_v<Op>) {
            // Dynamic rank expression — dispatch to concrete rank instantiation
            const int r = op.DynRank();
            switch (r) {
              case 0: ExecWithRank<0>(op); break;
              case 1: ExecWithRank<1>(op); break;
              case 2: ExecWithRank<2>(op); break;
              case 3: ExecWithRank<3>(op); break;
              case 4: ExecWithRank<4>(op); break;
              case 5: ExecWithRank<5>(op); break;
              case 6: ExecWithRank<6>(op); break;
              case 7: ExecWithRank<7>(op); break;
              case 8: ExecWithRank<8>(op); break;
              default:
                MATX_THROW(matxInvalidParameter,
                  "Dynamic rank " + std::to_string(r) + " exceeds MATX_MAX_DYNAMIC_RANK (" +
                  std::to_string(MATX_MAX_DYNAMIC_RANK) + ")");
            }
          } else {
            // Static rank path — zero overhead
            ExecWithRank<Op::Rank()>(op);
          }
#else
          MATX_ASSERT_STR(false, matxInvalidParameter, "Cannot call device executor using host compiler");
#endif
#else
          MATX_THROW(matxInvalidParameter, "JIT compilation is not enabled. Define MATX_EN_JIT to use CUDAJit executor.");
#endif
        }

    private:
      /**
       * Rank-parameterized execution. RANK is always a concrete compile-time value.
       * For static expressions it equals Op::Rank(); for dynamic expressions it is
       * the runtime rank resolved by the switch in Exec().
       */
      template <int RANK, typename Op>
        void ExecWithRank(const Op &op) const {
#ifdef MATX_EN_JIT
#ifdef __CUDACC__
          dim3 threads = 1;
          dim3 blocks = 1;

          // Parameters passed by value in CUDA are limited to CUDA_MAX_VAL_PARAM
          if ((sizeof(op) + sizeof(index_t) * RANK) > detail::CUDA_MAX_VAL_PARAM) {
            MATX_THROW(matxInvalidParameter,
                "Parameter buffer to device is limited to " + std::to_string(detail::CUDA_MAX_VAL_PARAM) + "B. "
                "Please break up your operator statement into multiple executions to limit the size of the parameters");
          }

          // The dynamic-rank dispatch in Exec() instantiates ExecWithRank for
          // every switch arm, so this template is compiled for RANK values
          // that exceed Op::Rank() (dead arms — never reached at runtime,
          // since the dynamic source's runtime rank must match Op::Rank()).
          // Cap both copies at min(RANK, Op::Rank()) so the dead instantiations
          // don't read past op.Size() bounds or write past op_sizes, which
          // trips -Werror=aggressive-loop-optimizations on older GCC. The
          // unfilled slots stay zero from `{}`-init; in the live arm this is
          // a no-op (RANK == Op::Rank()), and in the genuine dynamic case
          // (Op::Rank() > RANK) the trailing zeros were already the intent.
          constexpr int kSizeCount =
              (RANK < static_cast<int>(Op::Rank())) ? RANK : static_cast<int>(Op::Rank());

          cuda::std::array<index_t, RANK> sizes{};
          for (int i = 0; i < kSizeCount; i++) {
            sizes[i] = op.Size(i);
          }

          // create_kernel_provider requires an Op::Rank()-sized array.
          cuda::std::array<index_t, Op::Rank()> op_sizes{};
          for (int i = 0; i < kSizeCount; i++) {
            op_sizes[i] = sizes[i];
          }

          if constexpr (RANK <= 4) {
            const auto jit_cache_key = detail::get_operator_capability<detail::OperatorCapability::JIT_CACHE_KEY>(op);
            std::optional<std::string> kernel_op_type;
            auto ensure_kernel_op_type = [&]() -> const std::string& {
              if (!kernel_op_type.has_value()) {
                kernel_op_type = detail::get_operator_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(op);
              }
              return *kernel_op_type;
            };

            // Check if we have cached launch parameters for this operator type
            detail::JITLaunchParams cached_params;
            bool has_cached_params = jit_cache_key.valid ?
                detail::LookupJITLaunchParams(jit_cache_key, cached_params) :
                detail::LookupJITLaunchParams(ensure_kernel_op_type(), cached_params);

            detail::ElementsPerThread best_ept = detail::ElementsPerThread::INVALID;
            int shm_size = 0;
            int block_size = 0;
            int groups_per_block = 1;
            bool stride = false;
            bool global_kernel = false;
            bool pass_through_threads = false;
            int pass_through_inner_rank = 2;
            bool block_reduces_rank = false;
            detail::JITLaunchParams params_to_cache{};

            if (has_cached_params) {
              // Use cached parameters and skip capability/resource queries. These
              // queries can call libmathdx for MathDx-backed operators.
              MATX_LOG_DEBUG("Using cached launch parameters for JIT operator");
              best_ept = cached_params.best_ept;
              shm_size = cached_params.shm_size;
              block_size = cached_params.block_size;
              groups_per_block = cached_params.groups_per_block;
              stride = cached_params.stride;
              blocks = cached_params.blocks;
              threads = cached_params.threads;
              global_kernel = cached_params.global_kernel;
              pass_through_threads = cached_params.pass_through_threads;
              pass_through_inner_rank = cached_params.pass_through_inner_rank;
              block_reduces_rank = cached_params.block_reduces_rank;

              MATX_LOG_DEBUG("Cached EPT {}, Shm size {}, Block size {}, Groups per block {}",
                             static_cast<int>(best_ept), shm_size, block_size, groups_per_block);
            } else {
              // No cached parameters - compute them
              MATX_LOG_DEBUG("No cached parameters found, computing launch parameters for JIT");

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

              global_kernel = detail::get_operator_capability<detail::OperatorCapability::GLOBAL_KERNEL>(op);
              pass_through_threads = detail::get_operator_capability<detail::OperatorCapability::PASS_THROUGH_THREADS>(op);
              pass_through_inner_rank = detail::get_operator_capability<detail::OperatorCapability::PASS_THROUGH_INNER_RANK>(op);
              block_reduces_rank = detail::get_operator_capability<detail::OperatorCapability::BLOCK_REDUCES_RANK>(op);
              if (global_kernel) {
                MATX_LOG_DEBUG("Operator operates on a global level");
              } else if (block_reduces_rank) {
                MATX_LOG_DEBUG("Operator uses a reduced-rank block collective");
              } else if (pass_through_threads) {
                MATX_LOG_DEBUG("Operator uses pass-through threads with inner rank {}", pass_through_inner_rank);
              } else {
                MATX_LOG_DEBUG("Operator operates on a block level");
              }

              if (pass_through_threads && !block_reduces_rank) {
                // For pass-through operators (e.g., MathDx), block dimensions are constrained by the operator.
                auto block_dim_range = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
                block_size = detail::SelectJITPassThroughBlockDim(block_dim_range);
                stride = detail::get_grid_dims_block_pass_through<RANK>(blocks, threads, sizes, block_size, pass_through_inner_rank);

                // EPT is 1 for 2D block operators - the operator handles elements internally
                best_ept = detail::ElementsPerThread::ONE;
                shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
                groups_per_block = 1;

                MATX_LOG_DEBUG("Block2D: EPT {}, Shm size {}, Block size {}",
                               static_cast<int>(best_ept), shm_size, block_size);
              } else if constexpr (is_dynamic_rank_op_v<Op>) {
                // Dynamic tensor expressions: pre-compiled kernels don't exist for this Op type,
                // so we cannot query register pressure. Use conservative defaults.
                best_ept = jit_ept_bounds[1];
                shm_size = detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op);
                block_size = 256;
                groups_per_block = 1;

                MATX_LOG_DEBUG("Dynamic tensor EPT {}, Shm size {}, Block size {}",
                               static_cast<int>(best_ept), shm_size, block_size);

                if (global_kernel) {
                  stride = detail::get_grid_dims<RANK>(blocks, threads, sizes, static_cast<int>(best_ept), 256);
                } else if (block_reduces_rank) {
                  stride = detail::get_grid_dims_block_reduce<RANK>(blocks, threads, sizes, groups_per_block, block_size);
                } else {
                  stride = detail::get_grid_dims_block<RANK>(blocks, threads, sizes, static_cast<int>(best_ept), groups_per_block, block_size, true);
                }
              } else {
                // Create kernel provider for JIT using consolidated function
                auto kernel_provider = detail::create_kernel_provider<Op>(op_sizes, true, global_kernel);

                // Find the best launch parameters
                auto result = detail::find_best_launch_params(op, kernel_provider, 0, true);
                best_ept = cuda::std::get<0>(result);
                shm_size = cuda::std::get<1>(result);
                block_size = cuda::std::get<2>(result);
                groups_per_block = cuda::std::get<3>(result);

                MATX_LOG_DEBUG("Best EPT {}, Shm size {}, Block size {}, Groups per block {}",
                               static_cast<int>(best_ept), shm_size, block_size, groups_per_block);

                if (global_kernel) {
                  stride = detail::get_grid_dims<RANK>(blocks, threads, sizes, static_cast<int>(best_ept), 256);
                } else if (block_reduces_rank) {
                  stride = detail::get_grid_dims_block_reduce<RANK>(blocks, threads, sizes, groups_per_block, block_size);
                } else {
                  stride = detail::get_grid_dims_block<RANK>(blocks, threads, sizes, static_cast<int>(best_ept), groups_per_block, block_size, true);
                }
              }

              // Cache ALL parameters for future use (sizes are encoded in type string)
              params_to_cache.best_ept = best_ept;
              params_to_cache.shm_size = shm_size;
              params_to_cache.block_size = block_size;
              params_to_cache.groups_per_block = groups_per_block;
              params_to_cache.stride = stride;
              params_to_cache.blocks = blocks;
              params_to_cache.threads = threads;
              params_to_cache.osize = RANK == 0 ? 1 : static_cast<int>(op.Size(RANK - 1));
              params_to_cache.global_kernel = global_kernel;
              params_to_cache.pass_through_threads = pass_through_threads;
              params_to_cache.pass_through_inner_rank = pass_through_inner_rank;
              params_to_cache.block_reduces_rank = block_reduces_rank;
            }

            MATX_LOG_DEBUG("Shm size {}, Stride {}, estimated EPT {}, blocks {}x{}x{} threads {}x{}x{}, pass_through_threads {}",
                shm_size, stride, static_cast<int>(best_ept), blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, pass_through_threads);
            const int osize = has_cached_params ? cached_params.osize :
                (RANK == 0 ? 1 : static_cast<int>(op.Size(RANK - 1)));
            detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, shm_size, osize,
                                          global_kernel, stream_, pass_through_threads, pass_through_inner_rank,
                                          block_reduces_rank, jit_cache_key, kernel_op_type);
            if (!has_cached_params) {
              if (jit_cache_key.valid) {
                detail::StoreJITLaunchParams(jit_cache_key, params_to_cache);
              } else {
                detail::StoreJITLaunchParams(ensure_kernel_op_type(), params_to_cache);
              }
            }
          }
          else {
            // ND kernel support for ranks > 4 (JIT path)
            const auto jit_cache_key = detail::get_operator_capability<detail::OperatorCapability::JIT_CACHE_KEY>(op);
            std::optional<std::string> kernel_op_type;
            auto ensure_kernel_op_type = [&]() -> const std::string& {
              if (!kernel_op_type.has_value()) {
                kernel_op_type = detail::get_operator_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(op);
              }
              return *kernel_op_type;
            };

            // Check if we have cached launch parameters for this operator type
            detail::JITLaunchParams cached_params;
            bool has_cached_params = jit_cache_key.valid ?
                detail::LookupJITLaunchParams(jit_cache_key, cached_params) :
                detail::LookupJITLaunchParams(ensure_kernel_op_type(), cached_params);

            detail::ElementsPerThread best_ept = detail::ElementsPerThread::INVALID;
            bool stride = false;
            bool pass_through_threads = false;
            int pass_through_inner_rank = 2;
            detail::JITLaunchParams params_to_cache{};

            if (has_cached_params) {
              // Use cached parameters and skip capability/resource queries.
              MATX_LOG_DEBUG("Using cached launch parameters for ND JIT kernel");
              best_ept = cached_params.best_ept;
              stride = cached_params.stride;
              blocks = cached_params.blocks;
              threads = cached_params.threads;
              pass_through_threads = cached_params.pass_through_threads;
              pass_through_inner_rank = cached_params.pass_through_inner_rank;
            } else {
              // No cached parameters - compute them
              MATX_LOG_DEBUG("No cached parameters found, computing launch parameters for ND kernel");

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

              pass_through_threads = detail::get_operator_capability<detail::OperatorCapability::PASS_THROUGH_THREADS>(op);
              pass_through_inner_rank = detail::get_operator_capability<detail::OperatorCapability::PASS_THROUGH_INNER_RANK>(op);
              best_ept = jit_ept_bounds[1];
              stride = detail::get_grid_dims<RANK>(blocks, threads, sizes, static_cast<int>(best_ept), 1024);

              // Cache ALL parameters for future use (sizes are encoded in type string)
              params_to_cache.best_ept = best_ept;
              params_to_cache.shm_size = 0;
              params_to_cache.block_size = threads.x;
              params_to_cache.groups_per_block = 1;
              params_to_cache.stride = stride;
              params_to_cache.blocks = blocks;
              params_to_cache.threads = threads;
              params_to_cache.osize = RANK == 0 ? 1 : static_cast<int>(op.Size(RANK - 1));
              params_to_cache.global_kernel = true;
              params_to_cache.pass_through_threads = pass_through_threads;
              params_to_cache.pass_through_inner_rank = pass_through_inner_rank;
              params_to_cache.block_reduces_rank = false;
            }

            MATX_LOG_DEBUG("Using ND kernel for rank > 4 with JIT and EPT {}", static_cast<int>(best_ept));
            index_t dims = cuda::std::accumulate(cuda::std::begin(sizes) + 1, cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
            const int osize = has_cached_params ? cached_params.osize :
                (RANK == 0 ? 1 : static_cast<int>(op.Size(RANK - 1)));

            MATX_LOG_DEBUG("ND kernel: stride {}, blocks {}x{}x{} threads {}x{}x{}, dims {}",
                stride, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, dims);

            // Use ND kernel through JIT compilation
            detail::nvrtc_compile_and_run("output.cu", op, sizes, blocks, threads, best_ept, stride, 0, osize, true,
                                          stream_, pass_through_threads, pass_through_inner_rank,
                                          false, jit_cache_key, kernel_op_type);
            if (!has_cached_params) {
              if (jit_cache_key.valid) {
                detail::StoreJITLaunchParams(jit_cache_key, params_to_cache);
              } else {
                detail::StoreJITLaunchParams(ensure_kernel_op_type(), params_to_cache);
              }
            }
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
