////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include "matx.h"
#include <nvbench/nvbench.cuh>
#include "matx/core/nvtx.h"

#include <string>
#include <vector>

using namespace matx;

// Compares the cost of running dense solvers with and without a SolverStatus.
//
// Each benchmark iteration enqueues `calls` back-to-back solver calls on one
// stream and is timed until the work is done and, in every mode except
// `device`, the host has checked every result:
//
//   none    No status (baseline). Each call synchronizes the stream and checks
//           its info values on the host before returning.
//   device  User-provided status tensor in device memory, which can only be read
//           on the device. No host read; the iteration ends with one stream
//           synchronize. This is a lower bound, not equivalent work: the host
//           never learns whether a call failed.
//   pinned  MatX-owned status in host-pinned memory (the default), written
//           directly by the library through its mapped pointer; the iteration
//           ends with one stream synchronize, then check() on every status.
//
// With calls = 1 the result shows the per-call overhead of each mode. With
// calls > 1 it also shows how much of the baseline cost is the per-call sync.

using solver_status_types = nvbench::type_list<float, cuda::std::complex<double>>;

namespace {

enum class StatusMode { None, Device, Pinned };

StatusMode ParseStatusMode(const std::string &s)
{
  if (s == "device") {
    return StatusMode::Device;
  }
  if (s == "pinned") {
    return StatusMode::Pinned;
  }
  return StatusMode::None;
}

template <typename AType>
std::vector<SolverStatus> MakeStatuses(const AType &A, cudaExecutor &exec, StatusMode mode, index_t calls)
{
  std::vector<SolverStatus> statuses;
  if (mode == StatusMode::None) {
    return statuses;
  }

  for (index_t c = 0; c < calls; c++) {
    switch (mode) {
      case StatusMode::Device: {
        auto info = make_tensor<int>({2, A.Size(0)}, MATX_DEVICE_MEMORY);
        statuses.push_back(make_solver_status(A, info, exec));
        break;
      }
      case StatusMode::Pinned:
        statuses.push_back(make_solver_status(A, exec));
        break;
      default:
        break;
    }
  }
  return statuses;
}

// `solve(exec, status)` runs one solver call; status is nullptr for the baseline
template <typename AType, typename Solve>
void RunSolverStatusBench(nvbench::state &state, StatusMode mode, const AType &A, index_t calls,
                          Solve &&solve)
{
  cudaStream_t stream;
  MATX_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
  {
    // A CUDA status is created for the device of the stream nvbench launches on
    cudaExecutor setup{stream};
    const auto statuses = MakeStatuses(A, setup, mode, calls);
    setup.sync();

    auto iteration = [&](cudaExecutor &exec) {
      for (index_t c = 0; c < calls; c++) {
        solve(exec, mode == StatusMode::None ? nullptr : &statuses[c]);
      }

      if (mode != StatusMode::None) {
        exec.sync();
      }
      if (mode == StatusMode::Pinned) {
        for (const auto &st : statuses) {
          st.check();
        }
      }
    };

    // Warm up: creates the solver plans
    {
      cudaExecutor exec{stream};
      iteration(exec);
      exec.sync();
    }

    // The baseline synchronizes inside each call, so every mode uses sync timing
    MATX_NVTX_START_RANGE("Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1)
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
      cudaExecutor exec{launch.get_stream()};
      iteration(exec);
    });
    MATX_NVTX_END_RANGE(1)
  }

  // Plans are cached per stream and free their workspaces on it, so release
  // them before the stream
  ClearCaches();
  MATX_CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

template <typename ValueType>
void qr_solver_status(nvbench::state &state, nvbench::type_list<ValueType>)
{
  const index_t batch = static_cast<index_t>(state.get_int64("batch"));
  const index_t n = static_cast<index_t>(state.get_int64("cols"));
  const index_t m = 2 * n;
  const index_t calls = static_cast<index_t>(state.get_int64("calls"));
  const auto mode = ParseStatusMode(state.get_string("status"));

  cudaExecutor setup{};
  auto A = make_tensor<ValueType>({batch, m, n});
  auto out = make_tensor<ValueType>({batch, m, n});
  auto tau = make_tensor<ValueType>({batch, n});
  (A = random<ValueType>(A.Shape(), NORMAL)).run(setup);
  setup.sync();

  RunSolverStatusBench(state, mode, A, calls,
    [&](cudaExecutor &exec, const SolverStatus *st) {
      if (st == nullptr) {
        (mtie(out, tau) = qr_solver(A)).run(exec);
      }
      else {
        (mtie(out, tau) = qr_solver(A, *st)).run(exec);
      }
    });
}
NVBENCH_BENCH_TYPES(qr_solver_status, NVBENCH_TYPE_AXES(solver_status_types))
  .add_int64_axis("cols", {16, 128})
  .add_int64_axis("batch", {1, 32, 256})
  .add_int64_axis("calls", {1, 8})
  .add_string_axis("status", {"none", "device", "pinned"});

template <typename ValueType>
void qr_econ_status(nvbench::state &state, nvbench::type_list<ValueType>)
{
  const index_t batch = static_cast<index_t>(state.get_int64("batch"));
  const index_t n = static_cast<index_t>(state.get_int64("cols"));
  const index_t m = 2 * n;
  const index_t calls = static_cast<index_t>(state.get_int64("calls"));
  const auto mode = ParseStatusMode(state.get_string("status"));

  cudaExecutor setup{};
  auto A = make_tensor<ValueType>({batch, m, n});
  auto Q = make_tensor<ValueType>({batch, m, n});
  auto R = make_tensor<ValueType>({batch, n, n});
  (A = random<ValueType>(A.Shape(), NORMAL)).run(setup);
  setup.sync();

  RunSolverStatusBench(state, mode, A, calls,
    [&](cudaExecutor &exec, const SolverStatus *st) {
      if (st == nullptr) {
        (mtie(Q, R) = qr_econ(A)).run(exec);
      }
      else {
        (mtie(Q, R) = qr_econ(A, *st)).run(exec);
      }
    });
}
NVBENCH_BENCH_TYPES(qr_econ_status, NVBENCH_TYPE_AXES(solver_status_types))
  .add_int64_axis("cols", {16, 128})
  .add_int64_axis("batch", {1, 32, 256})
  .add_int64_axis("calls", {1, 8})
  .add_string_axis("status", {"none", "device", "pinned"});
