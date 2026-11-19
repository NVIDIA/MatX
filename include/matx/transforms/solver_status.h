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



#pragma once

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "matx/core/allocator.h"
#include "matx/core/capabilities.h"
#include "matx/core/error.h"
#include "matx/core/make_tensor.h"
#include "matx/core/resource_guard.h"
#include "matx/core/tensor.h"

namespace matx {

/**
 * @brief Options for make_solver_status().
 *
 * Aggregate with designated-initializer support. Fields left out keep their
 * defaults, e.g. `make_solver_status(A, exec, {.stages = 1})`.
 */
struct SolverStatusParams {
  /// Stages to allocate. The default, 2, covers every solver that accepts a
  /// status.
  int stages = 2;
};

namespace detail {

/**
 * Value make_solver_status fills a MatX-allocated status tensor with, so a read
 * after the first solver call is queued but before it writes the status reports
 * a failure instead of the success that zeros would. No library returns it.
 */
inline constexpr int solver_info_not_written = std::numeric_limits<int>::min();

/**
 * Text for one nonzero `info` value. A negative value names the illegal
 * parameter, which means the same in every LAPACK-style routine; a positive
 * value's meaning depends on the routine, so it is given as is.
 */
inline std::string DescribeSolverInfo(int info)
{
  if (info == solver_info_not_written) {
    return "no solver result written yet; synchronize after the solver call before reading the status";
  }
  // INT_MIN returned above, so negating cannot overflow
  return info < 0 ? "parameter " + std::to_string(-info) + " had an illegal value"
                  : "info = " + std::to_string(info);
}

/** How solvers reach a status tensor */
struct SolverInfoMemory {
  int *library_info = nullptr;   ///< Pointer the solver libraries write `info` through
  bool device_only = false;      ///< In device memory, so the values can only be read on the device
};

/**
 * Whether `data` is in device memory, which host solvers cannot write. Memory
 * MatX allocated is classified from MatX's own records. Other memory is
 * classified by the driver, which needs no current context and, before CUDA is
 * initialized, answers without initializing it: no device memory can exist
 * then. So a host status never starts CUDA or creates a context, and the
 * runtime's last error is left alone.
 */
inline bool IsDeviceOnlyMemory(void *data)
{
  const auto kind = GetPointerKind(data);
  if (kind == MATX_DEVICE_MEMORY || kind == MATX_ASYNC_DEVICE_MEMORY) {
    return true;
  }
  if (kind != MATX_INVALID_MEMORY) {
    return false;
  }
  CUmemorytype type{};
  int managed = 0;
  CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_IS_MANAGED};
  void *values[] = {&type, &managed};
  if (cuPointerGetAttributes(2, attrs, values, reinterpret_cast<CUdeviceptr>(data)) != CUDA_SUCCESS) {
    // CUDA is not initialized, or the memory is not CUDA's: ordinary host memory
    return false;
  }
  // Managed memory also reports CU_MEMORYTYPE_DEVICE, but the host can access it
  return type == CU_MEMORYTYPE_DEVICE && managed == 0;
}

/** The kind of an executor. Solver statuses support only Host and Cuda. */
enum class ExecutorKind { Host, Cuda, CudaJit, Distributed };

/** The ExecutorKind of `Executor`. CUDAJITExecutor is also a CUDA executor, so it is checked first. */
template <typename Executor>
constexpr ExecutorKind ExecutorKindOf()
{
  if constexpr (is_cuda_jit_executor_v<Executor>) {
    return ExecutorKind::CudaJit;
  } else if constexpr (is_distributed_executor_v<Executor>) {
    return ExecutorKind::Distributed;
  } else if constexpr (is_cuda_executor_v<Executor>) {
    return ExecutorKind::Cuda;
  } else {
    static_assert(is_host_executor_v<Executor>, "Unknown executor type");
    return ExecutorKind::Host;
  }
}

/** The ExecutorKind of `Executor` */
template <typename Executor>
inline constexpr ExecutorKind executor_kind_v = ExecutorKindOf<Executor>();

/** A name for an executor of `kind`, for messages */
inline const char *ExecutorKindName(ExecutorKind kind)
{
  switch (kind) {
    case ExecutorKind::Host: return "host executor";
    case ExecutorKind::Cuda: return "cudaExecutor";
    case ExecutorKind::CudaJit: return "CUDAJITExecutor";
    case ExecutorKind::Distributed: return "distributedCUDAExecutor";
  }
  return "unknown executor";
}

/**
 * Throw unless statuses can be created for executors of `kind`: host executors
 * and cudaExecutor. Every status therefore has one of those kinds, so using it
 * with any other executor is a mismatch.
 */
inline void RequireSolverStatusExecutor(ExecutorKind kind)
{
  if (kind != ExecutorKind::Host && kind != ExecutorKind::Cuda) {
    MATX_THROW(matxInvalidExecutor, std::string("Solver statuses support host executors and cudaExecutor, not a ") +
               ExecutorKindName(kind));
  }
}

/**
 * Classify the status tensor at `data` and check that solvers of the executor
 * kind it was created for can write it, so a status that no solver can use is
 * rejected at creation rather than at its first solver call.
 *
 * - A CUDA status needs device memory on `device`, the device of the creating
 *   executor's stream, passed as is, or host-pinned memory, passed through its
 *   device-mapped pointer. CUDA reports that pointer as null when the device
 *   cannot access the memory, such as on a device that cannot map host memory.
 *   Pageable and managed memory throw.
 * - A host status needs host-accessible memory, which host solvers write
 *   directly. Device memory throws.
 *
 * A CUDA status is classified on `device`, since a host-pinned tensor's mapped
 * pointer is reported for the device current at the query. CUDA statuses
 * always query the runtime, which also sees host memory the user has since
 * pinned with cudaHostRegister.
 */
inline SolverInfoMemory ResolveSolverInfoMemory(int *data, ExecutorKind kind, int device)
{
  if (kind == ExecutorKind::Host) {
    if (IsDeviceOnlyMemory(data)) {
      MATX_THROW(matxInvalidParameter,
                 "Host solvers write the status on the host, so a host executor's status tensor must be in host-accessible memory (pageable, host-pinned, or managed), not device memory");
    }
    return {data, false};
  }
  CudaDeviceGuard guard(device);
  cudaPointerAttributes attrs{};
  if (cudaPointerGetAttributes(&attrs, data) != cudaSuccess) {
    // Memory CUDA does not know about: ordinary host memory. The failure has
    // already replaced any earlier last error, so clearing it only keeps this
    // query's failure from reaching the caller.
    cudaGetLastError();
    attrs.type = cudaMemoryTypeUnregistered;
  }

  switch (attrs.type) {
    case cudaMemoryTypeDevice:
      if (attrs.device != device) {
        MATX_THROW(matxInvalidParameter, "Solver status tensor is on device " + std::to_string(attrs.device) +
                   ", but the executor's stream is on device " + std::to_string(device));
      }
      return {data, true};
    case cudaMemoryTypeHost:
      if (attrs.devicePointer == nullptr) {
        MATX_THROW(matxInvalidParameter,
                   "Solver status: the device cannot access the host-pinned status tensor through a mapped pointer; pass a status tensor in device memory to make_solver_status");
      }
      return {static_cast<int *>(attrs.devicePointer), false};
    case cudaMemoryTypeUnregistered:
      MATX_THROW(matxInvalidParameter,
                 "Solver status: CUDA solvers need the status tensor in device memory or host-pinned memory, not pageable memory");
    case cudaMemoryTypeManaged:
    default:
      MATX_THROW(matxInvalidParameter,
                 "Solver status: CUDA solvers need the status tensor in device memory or host-pinned memory, not managed memory");
  }
  return {};
}

/** Shared state behind a SolverStatus handle, and the logic behind its members */
struct SolverStatusState {
  ::matx::tensor_t<int, 2> info;      ///< [stages, batches] view of the status tensor
  ExecutorKind kind;                  ///< Kind of executor the status was created for
  SolverInfoMemory memory;
  /// Number of stages every solver call on this status writes, fixed by its
  /// first call so that reads cover the same stages whichever call, or graph
  /// replay, ran last. Zero before the first call, so reads then cover no values.
  index_t stages_used = 0;

  /** `device` is the CUDA status's device, used to classify its memory; ignored for a host status */
  SolverStatusState(::matx::tensor_t<int, 2> t, ExecutorKind k, int device) :
    info(std::move(t)),
    kind(k),
    memory(ResolveSolverInfoMemory(info.Data(), kind, device))
  {
  }
  SolverStatusState(const SolverStatusState &) = delete;
  SolverStatusState &operator=(const SolverStatusState &) = delete;

  index_t Stages() const { return info.Size(0); }
  index_t Batches() const { return info.Size(1); }

  // ---- User-facing logic, forwarded by SolverStatus

  /** Throw unless the status was created for `caller`, the kind of executor using it */
  void RequireMatchingExecutor(ExecutorKind caller) const
  {
    if (caller != kind) {
      MATX_THROW(matxInvalidExecutor, std::string("Solver status was created with a ") + ExecutorKindName(kind) +
                 ", so it cannot be used with a " + ExecutorKindName(caller));
    }
  }

  /** Throw if the status tensor is in device memory, which the host cannot read */
  void RequireHostReadable() const
  {
    if (memory.device_only) {
      MATX_THROW(matxInvalidParameter,
                 "This solver status tensor is in device memory, so its values can only be read on the device through infos(); use a host-pinned status tensor to read them on the host");
    }
  }

  /**
   * The values of the stages the last solver call wrote, stage-major: none
   * before the first call. Reads without waiting. The span points into the
   * host-readable status tensor, which make_solver_status requires to be
   * contiguous; a later solver call on this status overwrites it.
   */
  std::span<const int> HostValues() const
  {
    RequireHostReadable();
    const auto count = static_cast<size_t>(stages_used * Batches());
    return {info.Data(), count};
  }

  bool Ok() const
  {
    for (const int v : HostValues()) {
      if (v != 0) {
        return false;
      }
    }
    return true;
  }

  void Check() const
  {
    const auto values = HostValues();
    // matxException messages are limited to a few hundred characters
    constexpr int max_reported = 3;
    int failures = 0;
    std::string details;
    for (size_t i = 0; i < values.size(); i++) {
      if (values[i] == 0) {
        continue;
      }
      if (failures < max_reported) {
        details += "\n  stage " + std::to_string(static_cast<index_t>(i) / Batches()) + ", batch " +
                   std::to_string(static_cast<index_t>(i) % Batches()) + ": " + DescribeSolverInfo(values[i]);
      }
      failures++;
    }
    if (failures == 0) {
      return;
    }
    std::string msg = "Solver reported " + std::to_string(failures) + " failure" +
                      (failures == 1 ? "" : "s") + details;
    if (failures > max_reported) {
      msg += "\n  ... " + std::to_string(failures - max_reported) + " more";
    }
    MATX_THROW(matxSolverError, msg);
  }

  // ---- Solver-side logic, called by solvers through SolverStatus::state()

  /**
   * Validate a solver call that writes `stages` stages for `batches` matrices.
   * A status serves one solver: its first call that does not throw fixes the
   * stage count, and a call that writes a different count throws. Reads cover the fixed stages,
   * which every call writes, so they stay right when calls run in a different
   * order than the host made them, as when a captured graph is replayed after
   * a direct call.
   */
  void ValidateRun(index_t stages, index_t batches) const
  {
    if (stages > Stages()) {
      MATX_THROW(matxInvalidSize, "Solver status has " + std::to_string(Stages()) +
                 " stage" + (Stages() == 1 ? "" : "s") + ", but this solver needs " + std::to_string(stages));
    }
    if (batches != Batches()) {
      MATX_THROW(matxInvalidSize, "Solver status was created for " + std::to_string(Batches()) +
                 " matrices, but this solver call has " + std::to_string(batches));
    }
    if (stages_used != 0 && stages_used != stages) {
      MATX_THROW(matxInvalidParameter, "Solver status was first used by a solver that writes " +
                 std::to_string(stages_used) + " stage" + (stages_used == 1 ? "" : "s") +
                 ", so it cannot be used by one that writes " + std::to_string(stages) +
                 "; use a separate status for each solver");
    }
  }

  /**
   * @brief Validate the status for a solver call on a `caller` executor that
   * writes `stages` stages, and return the pointer the solver writes `info`
   * through: stage s of matrix i at `ptr + s * batches + i`. For a host
   * executor, this is the status tensor itself, which is host-accessible.
   */
  int *BeginRun(ExecutorKind caller, index_t stages, index_t batches)
  {
    RequireMatchingExecutor(caller);
    ValidateRun(stages, batches);
    stages_used = stages;
    return memory.library_info;
  }
};

} // namespace detail

/**
 * @brief Per-matrix error status for a solver call, checked without forcing a
 * host synchronization inside the call.
 *
 * Pass a status to a solver that accepts one: qr_solver(), qr_econ(),
 * chol(), lu(), eig(), svd(), inv(), or solve() with a dense A. The solver has
 * the library write each matrix's `info` value into the status tensor, queued
 * on the executor's stream like the solver's other outputs. The status does not track when those writes finish: order reads
 * after the solver call as for any other output, with ok(exec) or check(exec),
 * which synchronize the executor's stream first, or with your own stream or
 * event synchronization followed by ok() or check(). Reading too early returns
 * the values of an earlier call. A status tensor MatX allocated starts filled
 * with a value no library writes, so reading it too early after its first
 * call reports a failure; a user-provided tensor holds whatever it held, and
 * zeros there report success.
 *
 * ok() and check() read the status tensor on the host, which for that must be
 * in host-pinned memory (the default for a CUDA executor). A status tensor in
 * device memory is read only on the device, through infos().
 *
 * The status tensor has shape `[stages, batch...]`, where the batch dimensions
 * match the solver input and each stage holds the `info` values of one library
 * call per matrix. qr_solver(), chol(), lu(), eig(), and svd() use one stage;
 * qr_econ(), inv(), and solve() use two.
 *
 * A status created with a CUDA executor is used only by CUDA solvers, and one
 * created with a host executor only by host solvers. A CUDA status may be used
 * on any stream of its device, but calls on different streams must be ordered
 * by the caller so they do not write it at once, as for any output tensor. A
 * status tensor in device memory is specific to its device, as any device
 * tensor is: use it only with solvers on streams of that device. The device is
 * checked when the status is created, not at each solver call.
 *
 * A SolverStatus is a handle to shared state, so copies refer to the same
 * status. Reusing a status for another call overwrites the previous results.
 * A status, including its copies, must not be used from two threads at once.
 * As with any output tensor, the status must outlive the solver work that
 * writes it, including captured CUDA graphs that are launched later.
 *
 * A status serves one solver: its first call that does not throw fixes the
 * number of stages it holds, and a call to a solver that writes a different number, such as
 * qr_econ() (two stages) on a status first used by qr_solver() (one), throws
 * before queuing work. Use a separate status for each solver.
 *
 * Before any solver call there are no stages, so ok() returns true and check()
 * passes. Do not read a status for a solver call that threw: it may hold
 * partial values, an earlier call's values, or its initial contents, which
 * for a user-provided tensor of zeros report success. The next solver call
 * that does not throw replaces them. A solver call captured into a CUDA graph
 * writes the status when the graph launches, so order reads after the launch.
 */
class SolverStatus {
public:
  /** An empty status, which solvers treat as no status. Its members throw. */
  SolverStatus() = default;

  /** A status for existing shared state. make_solver_status() is the usual way to create one. */
  explicit SolverStatus(std::shared_ptr<detail::SolverStatusState> state) : state_(std::move(state)) {}

  /**
   * @brief Report whether every matrix succeeded, reading without waiting.
   *
   * The caller must have ordered this read after the solver call, for example
   * with exec.sync() or an event recorded after the call; otherwise the result
   * may not reflect it. Use ok(exec) to synchronize the executor first.
   *
   * @return true if every `info` value written by the last solver call is zero,
   * and true before any solver call
   * @throws matxInvalidParameter if the status tensor is in device memory
   */
  bool ok() const { return State().Ok(); }

  /**
   * @brief Synchronize `exec`, then report whether every matrix succeeded.
   *
   * For a CUDA executor this synchronizes its whole stream, including work
   * queued after the solver call; a host executor's solver calls finish before
   * returning, so it does not wait.
   *
   * @param exec Executor the solver ran on, of the kind the status was created with
   * @return As ok()
   * @throws matxInvalidParameter if the status tensor is in device memory, or
   * `exec` is not of the kind the status was created with
   */
  template <typename Executor>
    requires (is_executor_t<Executor>())
  bool ok(const Executor &exec) const
  {
    SyncWith(exec);
    return ok();
  }

  /**
   * @brief Throw matxSolverError if any matrix failed, reading without waiting.
   *
   * The caller must have ordered this read after the solver call, as for ok().
   * Use check(exec) to synchronize the executor first.
   *
   * The message gives the number of failures and names the first few, with
   * each one's stage, batch index, and `info` value. The batch index is the
   * matrix's linear index over the batch dimensions, in row-major order, which
   * is also its column in infos(). Use infos() to inspect every value. When MatX is built with
   * MATX_DISABLE_EXCEPTIONS, a failure logs the message and aborts instead;
   * use ok() in those builds.
   *
   * @throws matxSolverError if any `info` value is nonzero
   * @throws matxInvalidParameter if the status tensor is in device memory
   */
  void check() const { State().Check(); }

  /**
   * @brief Synchronize `exec` as ok(exec) does, then check() the status.
   *
   * @param exec Executor the solver ran on, of the kind the status was created with
   * @throws matxSolverError if any `info` value is nonzero
   * @throws matxInvalidParameter if the status tensor is in device memory, or
   * `exec` is not of the kind the status was created with
   */
  template <typename Executor>
    requires (is_executor_t<Executor>())
  void check(const Executor &exec) const
  {
    SyncWith(exec);
    check();
  }

  /**
   * @brief The status tensor, viewed as `[stages, batches]`, in its own memory
   * space.
   *
   * Reading it on the host requires host-pinned or other host-accessible memory
   * and must be ordered after the solver call. On the device, work queued on
   * the solver's stream after the call is ordered after the writes. Only the
   * stages written by the last solver call hold its values.
   */
  tensor_t<int, 2> infos() const { return State().info; }

  /**
   * @brief Direct access to the shared state, for advanced use.
   *
   * The state's fields and methods are implementation details and may change
   * between releases. Solvers receive this pointer, with nullptr meaning no
   * status.
   *
   * @return The state, or nullptr for a default-constructed (empty) status
   */
  detail::SolverStatusState *state() const { return state_.get(); }

private:
  detail::SolverStatusState &State() const
  {
    if (!state_) {
      MATX_THROW(matxInvalidParameter, "Empty solver status: create one with make_solver_status()");
    }
    return *state_;
  }

  template <typename Executor>
  void SyncWith(const Executor &exec) const
  {
    State().RequireMatchingExecutor(detail::executor_kind_v<Executor>);
    // Fail before waiting on the stream
    State().RequireHostReadable();
    if constexpr (is_cuda_executor_v<Executor>) {
      MATX_CUDA_CHECK(cudaStreamSynchronize(exec.getStream()));
    }
  }

  std::shared_ptr<detail::SolverStatusState> state_;
};

namespace detail {

/**
 * Number of matrices in a batched solver input (the product of all but the last two dimensions)
 */
template <typename TensorType>
__MATX_INLINE__ index_t GetNumBatches(const TensorType &a)
{
  index_t cnt = 1;
  for (int i = 3; i <= TensorType::Rank(); i++) {
    cnt *= a.Size(TensorType::Rank() - i);
  }

  return cnt;
}

/**
 * A status disables JIT, since a fused JIT kernel has no way to write it. An
 * operator that takes a status reports no JIT support with one, and also puts
 * the status into its JIT cache key and JIT type name: the JIT launch cache is
 * keyed by one or the other, and a cache hit skips the support check, so a
 * status operator must not match a cached no-status launch. The type name
 * marker is never compiled.
 */
__MATX_INLINE__ bool SolverStatusAllowsJit(const SolverStatusState *status)
{
  return status == nullptr;
}

/** Add whether there is a status to a JIT cache key. See SolverStatusAllowsJit(). */
__MATX_INLINE__ void HashSolverStatusJitKey(JITCacheKey &key, const SolverStatusState *status)
{
  HashJITCacheValue(key, status != nullptr);
}

/** Suffix for a JIT type name marking a status. See SolverStatusAllowsJit(). */
__MATX_INLINE__ const char *SolverStatusJitTypeSuffix(const SolverStatusState *status)
{
  return status != nullptr ? "/*status*/" : "";
}

/**
 * One solver call's use of a status, held for the whole call. Beginning the
 * call fixes the status's stage count. If the call then exits by an exception,
 * the destructor restores the previous count, so a first call that throws does
 * not tie the status to that solver.
 */
class SolverStatusRun {
public:
  /** No status */
  SolverStatusRun() = default;

  SolverStatusRun(SolverStatusState *state, int *info, index_t previous_stages) noexcept :
    state_(state), info_(info), previous_stages_(previous_stages), exceptions_(std::uncaught_exceptions())
  {
  }

  ~SolverStatusRun()
  {
    if (state_ != nullptr && std::uncaught_exceptions() > exceptions_) {
      state_->stages_used = previous_stages_;
    }
  }

  SolverStatusRun(const SolverStatusRun &) = delete;
  SolverStatusRun &operator=(const SolverStatusRun &) = delete;

  /** The pointer the solver writes `info` through, or nullptr without a status */
  int *Info() const noexcept { return info_; }

private:
  SolverStatusState *state_ = nullptr;
  int *info_ = nullptr;
  index_t previous_stages_ = 0;
  int exceptions_ = 0;
};

/**
 * Validate `status`, if any, for a solver call on a `caller` executor with
 * input `a` that writes `stages` stages. The result's Info() is the pointer
 * the solver writes `info` through, or nullptr without a status. Keep the
 * result alive until the solver call returns.
 */
template <typename OpA>
__MATX_INLINE__ SolverStatusRun BeginStatusRun(ExecutorKind caller, SolverStatusState *status, index_t stages, const OpA &a)
{
  if (status == nullptr) {
    return SolverStatusRun{};
  }
  const index_t previous = status->stages_used;
  int *info = status->BeginRun(caller, stages, GetNumBatches(a));
  return SolverStatusRun{status, info, previous};
}

/**
 * Check a solver input's batch dimensions, and call `check_dim(i)` for each
 * batch dimension i, such as to match a user status tensor against it. A
 * zero-size batch dimension would leave the status tensor with no memory to
 * write through, so it throws.
 */
template <typename OpA, typename CheckDim>
inline void ValidateSolverStatusBatches(const OpA &a, CheckDim &&check_dim)
{
  static_assert(OpA::Rank() >= 2, "Solver status input must have rank 2 or higher");
  for (int i = 0; i < OpA::Rank() - 2; i++) {
    if (a.Size(i) <= 0) {
      MATX_THROW(matxInvalidSize, "Solver status batch sizes must be positive");
    }
    check_dim(i);
  }
}

/** Shared state for a status tensor used with `exec` */
template <typename Executor>
inline std::shared_ptr<SolverStatusState> MakeSolverStatusState(::matx::tensor_t<int, 2> info,
                                                                 [[maybe_unused]] const Executor &exec)
{
  RequireSolverStatusExecutor(executor_kind_v<Executor>);
  int device = -1;
  if constexpr (is_cuda_executor_v<Executor>) {
    device = GetStreamDevice(exec.getStream());
  }
  return std::make_shared<SolverStatusState>(std::move(info), executor_kind_v<Executor>, device);
}

} // namespace detail

/**
 * @brief Create a solver status for an input shaped like `a`, for solvers on
 * `exec`'s kind of executor.
 *
 * MatX allocates a `[params.stages, batch...]` int tensor, filled with a value
 * no library writes. A read after the first solver call is queued but before
 * the call writes the status then reports that no result was written yet,
 * rather than success. Reads before any solver call cover no values.
 *
 * - With a CUDA executor, the tensor is host-pinned memory, which the library
 *   writes directly through its device-mapped pointer and the host reads
 *   directly. This requires a device that can map host memory, or creation
 *   throws; on such a device, pass a device-memory tensor to the other
 *   overload, whose values can then only be read on the device. Freeing the
 *   tensor with cudaFreeHost waits for all work on the device, so destroying
 *   the status does too; create a status once and reuse it rather than
 *   creating one per call.
 * - With a host executor, the tensor is ordinary host memory.
 *
 * @param a Solver input, used for its batch dimensions
 * @param exec Executor whose kind, and for CUDA whose stream's device, the
 * status is for
 * @param params Stage count
 */
template <typename OpA, typename Executor>
  requires (is_executor_t<Executor>())
SolverStatus make_solver_status(const OpA &a, const Executor &exec, const SolverStatusParams &params = {})
{
  static_assert(OpA::Rank() >= 2, "Solver status input must have rank 2 or higher");
  // Before allocating
  detail::RequireSolverStatusExecutor(detail::executor_kind_v<Executor>);
  if (params.stages < 1) {
    MATX_THROW(matxInvalidSize, "Solver status must have at least one stage");
  }
  detail::ValidateSolverStatusBatches(a, [](int) {});
  const cuda::std::array<index_t, 2> shape{static_cast<index_t>(params.stages), detail::GetNumBatches(a)};

  // Filled on the host before any solver call can be queued on it
  auto info = make_tensor<int>(shape, is_host_executor_v<Executor> ? MATX_HOST_MALLOC_MEMORY : MATX_HOST_MEMORY);
  std::fill_n(info.Data(), info.TotalSize(), detail::solver_info_not_written);
  return SolverStatus{detail::MakeSolverStatusState(info, exec)};
}

/**
 * @brief Create a solver status that writes into a user-provided tensor, for
 * solvers on `exec`'s kind of executor.
 *
 * The tensor must be a contiguous int tensor of shape `[stages, batch...]`; its
 * first dimension sets the stage count. MatX does not zero it.
 *
 * - With a CUDA executor, the tensor must be in host-pinned memory, which the
 *   device writes directly through its mapped pointer and which requires a
 *   device that can map host memory, or in device memory on the device of
 *   `exec`'s stream. A device-memory status is read only on the device, through
 *   infos(); ok() and check() throw for it. Other memory throws here.
 * - With a host executor, the tensor must be host-accessible: pageable,
 *   host-pinned, or managed memory, which host solvers write directly. Device
 *   memory throws here.
 *
 * The status keeps a reference to the tensor's storage.
 *
 * @param a Solver input, used for its batch dimensions
 * @param info Status tensor
 * @param exec Executor whose kind, and for CUDA whose stream's device, the
 * status is for
 */
template <typename OpA, typename InfoTensor, typename Executor>
  requires (is_tensor_view_v<InfoTensor> && is_executor_t<Executor>())
SolverStatus make_solver_status(const OpA &a, const InfoTensor &info, const Executor &exec)
{
  static_assert(std::is_same_v<typename InfoTensor::value_type, int>, "Solver status tensor must have value type int");
  static_assert(InfoTensor::Rank() == OpA::Rank() - 1,
                "Solver status tensor must have shape [stages, batch...], one rank lower than the solver input");

  detail::ValidateSolverStatusBatches(a, [&](int i) {
    if (info.Size(i + 1) != a.Size(i)) {
      MATX_THROW(matxInvalidSize, "Solver status tensor batch dimensions must match the solver input's batch dimensions");
    }
  });
  if (info.Size(0) < 1) {
    MATX_THROW(matxInvalidSize, "Solver status tensor must have at least one stage");
  }
  if (!info.IsContiguous()) {
    MATX_THROW(matxInvalidParameter, "Solver status tensor must be contiguous");
  }

  const index_t batches = detail::GetNumBatches(a);
  auto copy = info;
  return SolverStatus{detail::MakeSolverStatusState(copy.View({info.Size(0), batches}), exec)};
}

} // namespace matx
