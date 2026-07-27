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

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

} // namespace detail

/** Identifies a CUDA device in a process participating in distributed work. */
struct distributed_endpoint_t {
  int process_rank = 0;
  int device_id = 0;

  friend bool operator==(const distributed_endpoint_t &,
                         const distributed_endpoint_t &) = default;
};

/**
 * Lightweight topology identity shared by distributed tensors and executors.
 *
 * Communication-library handles intentionally do not live here.  A later
 * multi-process executor can attach MPI/NCCL state without changing the tensor
 * representation.
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

} // namespace detail

/**
 * Experimental single-process, multi-GPU executor.
 *
 * One non-blocking CUDA stream is owned per local endpoint.  Multi-process
 * collective support is deliberately left behind this executor boundary.
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
      : distributedCUDAExecutor(
            distributed_context{std::move(local_devices)}) {}

  distributedCUDAExecutor(const distributedCUDAExecutor &) = delete;
  distributedCUDAExecutor &
  operator=(const distributedCUDAExecutor &) = delete;

  distributedCUDAExecutor(distributedCUDAExecutor &&other) noexcept
      : context_{std::move(other.context_)},
        streams_{std::move(other.streams_)},
        transfer_count_{other.transfer_count_.load(std::memory_order_relaxed)} {
    other.streams_.clear();
  }

  distributedCUDAExecutor &operator=(distributedCUDAExecutor &&) = delete;

  ~distributedCUDAExecutor() { DestroyStreams(); }

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
  }

  void RecordTransfer() const noexcept {
    transfer_count_.fetch_add(1, std::memory_order_relaxed);
  }

  size_t TransferCount() const noexcept {
    return transfer_count_.load(std::memory_order_relaxed);
  }

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

  distributed_context context_;
  std::vector<stream_entry_t> streams_;
  mutable std::atomic<size_t> transfer_count_{0};
};

} // namespace matx
