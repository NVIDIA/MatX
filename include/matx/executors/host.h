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
#include <type_traits>
#include <cuda/std/array>

#include "matx/core/error.h"
#include "matx/core/get_grid_dims.h"
#ifdef MATX_EN_OMP
#include <omp.h>
#endif
namespace matx 
{

// Matches current Linux max
static constexpr int MAX_CPUS = 1024;
struct cpu_set_t {
  using set_type = uint64_t;

  cuda::std::array<set_type, MAX_CPUS / (8 * sizeof(set_type))> bits_;
};


struct HostExecParams {
  HostExecParams(int threads = 1) : threads_(threads) {}
  HostExecParams(cpu_set_t cpu_set) : cpu_set_(cpu_set), threads_(1) {
    MATX_ASSERT_STR(false, matxNotSupported, "CPU affinity not supported yet");
  }

  int GetNumThreads() const { return threads_; }

  private:
    int threads_;
    cpu_set_t cpu_set_;
};

/**
 * @brief Executor for running an operator on a single or multi-threaded host
 * 
 * @tparam MODE Threading policy
 * 
 */
template <ThreadsMode MODE = ThreadsMode::SINGLE>
class HostExecutor {
  public:
    using matx_cpu = bool; ///< Type trait indicating this is a CPU executor
    using matx_executor = bool; ///< Type trait indicating this is an executor

    HostExecutor() {
      int n_threads = 1;
      if constexpr (MODE == ThreadsMode::SINGLE) {
        n_threads = 1;
      }
      else if constexpr (MODE == ThreadsMode::ALL) {
#ifdef MATX_EN_OMP
        n_threads = omp_get_num_procs();
#endif
      }
      params_ = HostExecParams(n_threads);

#ifdef MATX_EN_OMP
      omp_set_num_threads(params_.GetNumThreads());
#endif
    }

    HostExecutor(const HostExecParams &params) : params_(params) {
#ifdef MATX_EN_OMP
      omp_set_num_threads(params_.GetNumThreads());
#endif
    }

    /**
     * @brief Synchronize the host executor's threads.
     * 
     */
    void sync() {}

    /**
     * @brief Execute an operator
     * 
     * @tparam Op Operator type
     * @param op Operator to execute
     */
    template <typename Op>
    void Exec(const Op &op) const noexcept {
      if constexpr (Op::Rank() == 0) {
        op();
      } 
      else {
        index_t size = TotalSize(op);
  #ifdef MATX_EN_OMP
        if (params_.GetNumThreads() > 1) {
          #pragma omp parallel for num_threads(params_.GetNumThreads())
          for (index_t i = 0; i < size; i++) {
            auto idx = GetIdxFromAbs(op, i);
            cuda::std::apply([&](auto... args) {
              return op(args...);
            }, idx);        
          }
        } else
  #endif
        { 
          for (index_t i = 0; i < size; i++) {
            auto idx = GetIdxFromAbs(op, i);
            cuda::std::apply([&](auto... args) {
              return op(args...);
            }, idx);        
          }      
        }
      }
    }

    int GetNumThreads() const { return params_.GetNumThreads(); }

    private:
      HostExecParams params_;
};

}
