////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(MATX_EN_NCCL) || defined(MATX_EN_CUBLASMP) ||                      \
    defined(MATX_EN_CUSOLVERMP)
#include <nccl.h>
#endif
#ifdef MATX_EN_CUBLASMP
#if __has_include(<cublasMp.h>)
#include <cublasMp.h>
#else
#include <cublasmp.h>
#endif
#endif
#ifdef MATX_EN_CUSOLVERMP
#include <cusolverMp.h>
#endif

#include "matx/core/error.h"
#include "matx/executors/cuda.h"

namespace matx {

namespace detail {

inline void DistributedCheck(bool condition, matxError_t error,
                             const char *message) {
  if (!condition) {
    MATX_THROW(error, message);
  }
}

inline void DistributedCheck(bool condition, matxError_t error,
                             const std::string &message) {
  if (!condition) {
    MATX_THROW(error, message);
  }
}

#ifdef MATX_EN_CUBLASMP
inline void CublasMpCheck(cublasMpStatus_t status, const char *operation) {
  if (status != CUBLASMP_STATUS_SUCCESS) {
    MATX_THROW(matxMatMulError, std::string(operation) +
                                    " failed with cuBLASMp status " +
                                    std::to_string(static_cast<int>(status)));
  }
}
#endif

#ifdef MATX_EN_CUSOLVERMP
inline void CusolverMpCheck(cusolverStatus_t status, const char *operation) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    MATX_THROW(matxSolverError, std::string(operation) +
                                    " failed with cuSOLVERMp status " +
                                    std::to_string(static_cast<int>(status)));
  }
}
#endif

#ifdef MATX_EN_NCCL
inline void NcclCheck(ncclResult_t status, const char *operation) {
  if (status != ncclSuccess) {
    MATX_THROW(matxInvalidExecutor, std::string(operation) + " failed: " +
                                        ncclGetErrorString(status));
  }
}
#endif

} // namespace detail

/** Identifies a CUDA device in a process participating in distributed work. */
struct distributed_endpoint_t {
  int process_rank = 0;
  int device_id = 0;

  friend bool operator==(const distributed_endpoint_t &,
                         const distributed_endpoint_t &) = default;
};

/** Ordering used to map communicator ranks onto a two-dimensional grid. */
enum class distributed_grid_layout { row_major, column_major };

#ifdef MATX_EN_NCCL
/** A borrowed NCCL communicator attached to one locally owned endpoint. */
struct distributed_nccl_binding_t {
  distributed_endpoint_t endpoint;
  ncclComm_t communicator = nullptr;
};

/**
 * Global NCCL rank ordering plus the communicators owned by this process.
 * MatX borrows the communicators; the caller must keep them alive until every
 * operation using the executor has completed.
 */
struct distributed_nccl_topology_t {
  std::vector<distributed_endpoint_t> rank_to_endpoint;
  std::vector<distributed_nccl_binding_t> local_communicators;
};
#endif

/**
 * Lightweight topology identity shared by distributed tensors and executors.
 *
 * Communication-library handles intentionally do not live here. An executor
 * can attach collective state without changing the tensor representation.
 */
class distributed_context {
private:
  struct state_t {
    uint64_t id;
    int process_rank;
    int process_count;
    std::vector<int> local_devices;
  };

  static uint64_t NextId() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
  }

public:
  explicit distributed_context(std::vector<int> local_devices,
                               int process_rank = 0, int process_count = 1)
      : state_{std::make_shared<state_t>(state_t{
            NextId(), process_rank, process_count, std::move(local_devices)})} {
    detail::DistributedCheck(process_count > 0, matxInvalidParameter,
                             "Distributed process count must be positive");
    detail::DistributedCheck(process_rank >= 0 && process_rank < process_count,
                             matxInvalidParameter,
                             "Invalid distributed process rank");
    detail::DistributedCheck(
        !state_->local_devices.empty(), matxInvalidParameter,
        "A distributed context needs at least one local CUDA device");

    for (size_t i = 0; i < state_->local_devices.size(); ++i) {
      detail::DistributedCheck(state_->local_devices[i] >= 0,
                               matxInvalidParameter,
                               "CUDA device IDs must be non-negative");
      for (size_t j = i + 1; j < state_->local_devices.size(); ++j) {
        detail::DistributedCheck(
            state_->local_devices[i] != state_->local_devices[j],
            matxInvalidParameter,
            "A CUDA device may appear only once in a distributed context");
      }
    }
  }

  uint64_t Id() const noexcept { return state_->id; }
  int ProcessRank() const noexcept { return state_->process_rank; }
  int ProcessCount() const noexcept { return state_->process_count; }
  const std::vector<int> &LocalDevices() const noexcept {
    return state_->local_devices;
  }

  bool IsLocalDevice(int device_id) const noexcept {
    for (int local_device : state_->local_devices) {
      if (local_device == device_id) {
        return true;
      }
    }
    return false;
  }

private:
  std::shared_ptr<const state_t> state_;
};

namespace detail {

class distributed_device_guard {
public:
  explicit distributed_device_guard(int device) {
    MATX_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) {
      MATX_CUDA_CHECK(cudaSetDevice(device));
      changed_ = true;
    }
  }

  distributed_device_guard(const distributed_device_guard &) = delete;
  distributed_device_guard &
  operator=(const distributed_device_guard &) = delete;

  ~distributed_device_guard() {
    if (changed_) {
      // Destructors must not throw.  A later CUDA call will report a failure to
      // restore the original device if the context has become unusable.
      (void)cudaSetDevice(previous_);
    }
  }

private:
  int previous_ = 0;
  bool changed_ = false;
};

#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
struct distributed_collective_state {
  ~distributed_collective_state() {
    int previous_device = 0;
    const bool restore_device =
        cudaGetDevice(&previous_device) == cudaSuccess && device_id >= 0;
    if (device_id >= 0) {
      (void)cudaSetDevice(device_id);
    }
#ifdef MATX_EN_CUSOLVERMP
    if (cusolver_grid != nullptr) {
      (void)cusolverMpDestroyGrid(cusolver_grid);
    }
    if (cusolver_handle != nullptr) {
      (void)cusolverMpDestroy(cusolver_handle);
    }
#endif
#ifdef MATX_EN_CUBLASMP
    if (cublas_grid != nullptr) {
      (void)cublasMpGridDestroy(cublas_grid);
    }
    if (cublas_handle != nullptr) {
      (void)cublasMpDestroy(cublas_handle);
    }
#endif
    if (restore_device) {
      (void)cudaSetDevice(previous_device);
    }
  }

  int device_id = -1;
  ncclComm_t communicator = nullptr;
  int process_rows = 0;
  int process_columns = 0;
  distributed_grid_layout grid_layout = distributed_grid_layout::row_major;
#ifdef MATX_EN_CUBLASMP
  cublasMpHandle_t cublas_handle = nullptr;
  cublasMpGrid_t cublas_grid = nullptr;
#endif
#ifdef MATX_EN_CUSOLVERMP
  cusolverMpHandle_t cusolver_handle = nullptr;
  cusolverMpGrid_t cusolver_grid = nullptr;
#endif
};
#endif

} // namespace detail

/**
 * Experimental distributed CUDA executor.
 *
 * One non-blocking CUDA stream is owned per local endpoint. When an NVIDIA MP
 * backend is enabled, an optional constructor attaches a borrowed NCCL
 * communicator and creates the backend handles needed for collective work.
 */
class distributedCUDAExecutor {
public:
  using matx_executor = bool;
  using distributed_executor = bool;

  explicit distributedCUDAExecutor(distributed_context context)
      : context_{std::move(context)} {
    try {
      streams_.reserve(context_.LocalDevices().size());
      for (int device : context_.LocalDevices()) {
        detail::distributed_device_guard guard{device};
        cudaStream_t stream{};
        MATX_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        streams_.push_back({device, stream});
      }
    } catch (...) {
      DestroyStreams();
      throw;
    }
  }

  explicit distributedCUDAExecutor(std::vector<int> local_devices)
      : distributedCUDAExecutor(distributed_context{std::move(local_devices)}) {
  }

#ifdef MATX_EN_NCCL
  distributedCUDAExecutor(distributed_context context,
                          distributed_nccl_topology_t topology)
      : distributedCUDAExecutor(std::move(context)) {
    try {
      InitializeNcclTopology(std::move(topology));
    } catch (...) {
      DestroyStreams();
      throw;
    }
  }
#endif

#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
  distributedCUDAExecutor(
      distributed_context context, ncclComm_t communicator, int process_rows,
      int process_columns,
      distributed_grid_layout layout = distributed_grid_layout::row_major)
      : distributedCUDAExecutor(std::move(context)) {
    try {
      InitializeCollectives(communicator, process_rows, process_columns,
                            layout);
    } catch (...) {
      DestroyStreams();
      throw;
    }
  }
#endif

  distributedCUDAExecutor(const distributedCUDAExecutor &) = delete;
  distributedCUDAExecutor &operator=(const distributedCUDAExecutor &) = delete;

  distributedCUDAExecutor(distributedCUDAExecutor &&other) noexcept
      : context_{std::move(other.context_)}, streams_{std::move(
                                                 other.streams_)},
        transfer_count_{other.transfer_count_.load(std::memory_order_relaxed)}
#ifdef MATX_EN_NCCL
        ,
        nccl_topology_{std::move(other.nccl_topology_)}
#endif
#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
        ,
        collectives_{std::move(other.collectives_)}
#endif
  {
    other.streams_.clear();
  }

  distributedCUDAExecutor &operator=(distributedCUDAExecutor &&) = delete;

  ~distributedCUDAExecutor() {
#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
    collectives_.reset();
#endif
    DestroyStreams();
  }

  const distributed_context &Context() const noexcept { return context_; }
  uint64_t ContextId() const noexcept { return context_.Id(); }

  cudaExecutor LocalExecutor(const distributed_endpoint_t &endpoint) const {
    detail::DistributedCheck(
        endpoint.process_rank == context_.ProcessRank(), matxInvalidExecutor,
        "The requested endpoint belongs to a different process");
    for (const auto &entry : streams_) {
      if (entry.device_id == endpoint.device_id) {
        return cudaExecutor{entry.stream};
      }
    }
    MATX_THROW(matxInvalidExecutor,
               "The requested device is not part of this distributed executor");
  }

  template <typename Callable>
  void ForEndpoint(const distributed_endpoint_t &endpoint,
                   Callable &&callable) const {
    auto local_executor = LocalExecutor(endpoint);
    detail::distributed_device_guard guard{endpoint.device_id};
    callable(local_executor);
  }

  void sync() {
    for (const auto &entry : streams_) {
      detail::distributed_device_guard guard{entry.device_id};
      MATX_CUDA_CHECK(cudaStreamSynchronize(entry.stream));
    }
#ifdef MATX_EN_NCCL
    if (nccl_topology_ != nullptr) {
      for (const auto &binding : nccl_topology_->local_communicators) {
        ncclResult_t asynchronous = ncclSuccess;
        detail::NcclCheck(
            ncclCommGetAsyncError(binding.communicator, &asynchronous),
            "ncclCommGetAsyncError");
        detail::NcclCheck(asynchronous, "NCCL asynchronous operation");
      }
    }
#endif
  }

  void RecordTransfer() const noexcept {
    transfer_count_.fetch_add(1, std::memory_order_relaxed);
  }

  size_t TransferCount() const noexcept {
    return transfer_count_.load(std::memory_order_relaxed);
  }

#ifdef MATX_EN_NCCL
  bool HasNcclTopology() const noexcept { return nccl_topology_ != nullptr; }

  const std::vector<distributed_endpoint_t> &CollectiveEndpoints() const {
    detail::DistributedCheck(HasNcclTopology(), matxInvalidExecutor,
                             "The distributed executor has no NCCL topology");
    return nccl_topology_->rank_to_endpoint;
  }

  ncclComm_t NcclCommunicator(const distributed_endpoint_t &endpoint) const {
    detail::DistributedCheck(HasNcclTopology(), matxInvalidExecutor,
                             "The distributed executor has no NCCL topology");
    for (const auto &binding : nccl_topology_->local_communicators) {
      if (binding.endpoint == endpoint) {
        return binding.communicator;
      }
    }
    MATX_THROW(matxInvalidExecutor,
               "No NCCL communicator is bound to the requested endpoint");
  }
#else
  bool HasNcclTopology() const noexcept { return false; }
#endif

#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
  bool HasCollectiveResources() const noexcept {
    return collectives_ != nullptr;
  }
  int ProcessRows() const noexcept { return collectives_->process_rows; }
  int ProcessColumns() const noexcept { return collectives_->process_columns; }
  distributed_grid_layout GridLayout() const noexcept {
    return collectives_->grid_layout;
  }
#else
  bool HasCollectiveResources() const noexcept { return false; }
#endif

#ifdef MATX_EN_CUBLASMP
  cublasMpHandle_t CublasHandle() const noexcept {
    return collectives_->cublas_handle;
  }
  cublasMpGrid_t CublasGrid() const noexcept {
    return collectives_->cublas_grid;
  }
#endif
#ifdef MATX_EN_CUSOLVERMP
  cusolverMpHandle_t CusolverHandle() const noexcept {
    return collectives_->cusolver_handle;
  }
  cusolverMpGrid_t CusolverGrid() const noexcept {
    return collectives_->cusolver_grid;
  }
#endif

private:
  struct stream_entry_t {
    int device_id;
    cudaStream_t stream;
  };

  void DestroyStreams() noexcept {
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    for (const auto &entry : streams_) {
      if (cudaSetDevice(entry.device_id) == cudaSuccess) {
        (void)cudaStreamDestroy(entry.stream);
      }
    }
    if (restore_device) {
      (void)cudaSetDevice(previous_device);
    }
    streams_.clear();
  }

#ifdef MATX_EN_NCCL
  void InitializeNcclTopology(distributed_nccl_topology_t topology) {
    detail::DistributedCheck(!topology.rank_to_endpoint.empty(),
                             matxInvalidParameter,
                             "An NCCL topology needs at least one endpoint");
    for (size_t i = 0; i < topology.rank_to_endpoint.size(); ++i) {
      const auto &endpoint = topology.rank_to_endpoint[i];
      detail::DistributedCheck(
          endpoint.process_rank >= 0 &&
              endpoint.process_rank < context_.ProcessCount(),
          matxInvalidParameter,
          "NCCL topology contains an invalid process rank");
      for (size_t j = i + 1; j < topology.rank_to_endpoint.size(); ++j) {
        detail::DistributedCheck(endpoint != topology.rank_to_endpoint[j],
                                 matxInvalidParameter,
                                 "NCCL topology endpoints must be unique");
      }
    }

    for (int device : context_.LocalDevices()) {
      const distributed_endpoint_t expected{context_.ProcessRank(), device};
      const auto rank_it = std::find(topology.rank_to_endpoint.begin(),
                                     topology.rank_to_endpoint.end(), expected);
      detail::DistributedCheck(
          rank_it != topology.rank_to_endpoint.end(), matxInvalidParameter,
          "Every local device must appear in the NCCL topology");
      const auto binding_it = std::find_if(
          topology.local_communicators.begin(),
          topology.local_communicators.end(),
          [&](const auto &binding) { return binding.endpoint == expected; });
      detail::DistributedCheck(
          binding_it != topology.local_communicators.end(),
          matxInvalidParameter,
          "Every local endpoint needs an NCCL communicator");
      detail::DistributedCheck(binding_it->communicator != nullptr,
                               matxInvalidParameter,
                               "NCCL communicators cannot be null");
      int communicator_count = 0;
      int communicator_rank = -1;
      detail::NcclCheck(
          ncclCommCount(binding_it->communicator, &communicator_count),
          "ncclCommCount");
      detail::NcclCheck(
          ncclCommUserRank(binding_it->communicator, &communicator_rank),
          "ncclCommUserRank");
      detail::DistributedCheck(
          communicator_count ==
                  static_cast<int>(topology.rank_to_endpoint.size()) &&
              communicator_rank ==
                  static_cast<int>(rank_it - topology.rank_to_endpoint.begin()),
          matxInvalidParameter,
          "NCCL communicator rank ordering does not match the endpoint "
          "topology");
    }
    detail::DistributedCheck(
        topology.local_communicators.size() == context_.LocalDevices().size(),
        matxInvalidParameter,
        "NCCL topology has duplicate or non-local communicator bindings");
    nccl_topology_ =
        std::make_unique<distributed_nccl_topology_t>(std::move(topology));
  }
#endif

#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
  void InitializeCollectives(ncclComm_t communicator, int process_rows,
                             int process_columns,
                             distributed_grid_layout layout) {
    detail::DistributedCheck(
        communicator != nullptr, matxInvalidExecutor,
        "Collective execution requires an NCCL communicator");
    detail::DistributedCheck(
        context_.LocalDevices().size() == 1, matxNotSupported,
        "NVIDIA MP execution currently supports one GPU per process");
    detail::DistributedCheck(
        process_rows > 0 && process_columns > 0 &&
            process_rows * process_columns == context_.ProcessCount(),
        matxInvalidSize,
        "The MP process grid must contain every distributed process");

    auto state = std::make_unique<detail::distributed_collective_state>();
    state->device_id = context_.LocalDevices().front();
    state->communicator = communicator;
    state->process_rows = process_rows;
    state->process_columns = process_columns;
    state->grid_layout = layout;
    const distributed_endpoint_t endpoint{context_.ProcessRank(),
                                          context_.LocalDevices().front()};
    auto local = LocalExecutor(endpoint);
    detail::distributed_device_guard guard{endpoint.device_id};

#ifdef MATX_EN_CUBLASMP
    detail::CublasMpCheck(
        cublasMpCreate(&state->cublas_handle, local.getStream()),
        "cublasMpCreate");
    detail::CublasMpCheck(
        cublasMpGridCreate(state->process_rows, state->process_columns,
                           state->grid_layout ==
                                   distributed_grid_layout::column_major
                               ? CUBLASMP_GRID_LAYOUT_COL_MAJOR
                               : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
                           state->communicator, &state->cublas_grid),
        "cublasMpGridCreate");
#endif

#ifdef MATX_EN_CUSOLVERMP
    detail::CusolverMpCheck(cusolverMpCreate(&state->cusolver_handle,
                                             endpoint.device_id,
                                             local.getStream()),
                            "cusolverMpCreate");
    detail::CusolverMpCheck(
        cusolverMpCreateDeviceGrid(
            state->cusolver_handle, &state->cusolver_grid, state->communicator,
            state->process_rows, state->process_columns,
            state->grid_layout == distributed_grid_layout::column_major
                ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR),
        "cusolverMpCreateDeviceGrid");
#endif
    collectives_ = std::move(state);
  }
#endif

  distributed_context context_;
  std::vector<stream_entry_t> streams_;
  mutable std::atomic<size_t> transfer_count_{0};
#ifdef MATX_EN_NCCL
  std::unique_ptr<distributed_nccl_topology_t> nccl_topology_;
#endif
#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
  std::unique_ptr<detail::distributed_collective_state> collectives_;
#endif
};

} // namespace matx
