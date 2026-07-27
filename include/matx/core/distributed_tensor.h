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

#include <cuda/std/numeric>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "matx/core/error.h"
#include "matx/core/make_tensor.h"
#include "matx/core/tensor.h"
#include "matx/executors/distributed.h"
#include "matx/operators/apply.h"

namespace matx {
namespace experimental {

template <int RANK> using distributed_index_t = cuda::std::array<index_t, RANK>;

namespace detail {

template <int RANK>
distributed_index_t<RANK>
DistributedUnflatten(index_t linear, const distributed_index_t<RANK> &shape) {
  distributed_index_t<RANK> index{};
  for (int dim = RANK - 1; dim >= 0; --dim) {
    index[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  return index;
}

template <int RANK>
index_t DistributedFlatten(const distributed_index_t<RANK> &index,
                           const distributed_index_t<RANK> &shape) {
  index_t linear = 0;
  for (int dim = 0; dim < RANK; ++dim) {
    linear = linear * shape[dim] + index[dim];
  }
  return linear;
}

} // namespace detail

/** Description of one rectangular fragment of a block distribution. */
template <int RANK> struct block_fragment_descriptor_t {
  distributed_endpoint_t endpoint;
  distributed_index_t<RANK> origin{};
  distributed_index_t<RANK> shape{};

  friend bool operator==(const block_fragment_descriptor_t &,
                         const block_fragment_descriptor_t &) = default;
};

/** A collection of non-overlapping rectangular global boxes. */
template <int RANK> class block_distribution_t {
public:
  static_assert(RANK > 0, "Distributed rank-zero tensors are not supported");
  static constexpr int Rank() noexcept { return RANK; }

  block_distribution_t(distributed_index_t<RANK> global_shape,
                       std::vector<block_fragment_descriptor_t<RANK>> fragments)
      : global_shape_{global_shape}, fragments_{std::move(fragments)} {
    Validate();
  }

  static block_distribution_t
  Slab(distributed_index_t<RANK> global_shape,
       const std::vector<distributed_endpoint_t> &endpoints, int axis = 0) {
    matx::detail::DistributedCheck(
        axis >= 0 && axis < RANK, matxInvalidDim,
        "Slab distribution axis is outside the tensor rank");
    matx::detail::DistributedCheck(
        !endpoints.empty(), matxInvalidParameter,
        "A slab distribution needs at least one endpoint");
    matx::detail::DistributedCheck(
        global_shape[axis] >= static_cast<index_t>(endpoints.size()),
        matxInvalidSize, "Slab distribution cannot create empty fragments");

    std::vector<block_fragment_descriptor_t<RANK>> fragments;
    fragments.reserve(endpoints.size());
    const index_t base =
        global_shape[axis] / static_cast<index_t>(endpoints.size());
    const index_t remainder =
        global_shape[axis] % static_cast<index_t>(endpoints.size());
    index_t offset = 0;
    for (size_t i = 0; i < endpoints.size(); ++i) {
      auto local_shape = global_shape;
      local_shape[axis] = base + (static_cast<index_t>(i) < remainder ? 1 : 0);
      distributed_index_t<RANK> origin{};
      origin[axis] = offset;
      fragments.push_back({endpoints[i], origin, local_shape});
      offset += local_shape[axis];
    }
    return block_distribution_t{global_shape, std::move(fragments)};
  }

  const distributed_index_t<RANK> &GlobalShape() const noexcept {
    return global_shape_;
  }

  size_t FragmentCount() const noexcept { return fragments_.size(); }

  const distributed_endpoint_t &FragmentEndpoint(size_t fragment) const {
    return fragments_.at(fragment).endpoint;
  }

  const distributed_index_t<RANK> &LocalShape(size_t fragment) const {
    return fragments_.at(fragment).shape;
  }

  distributed_index_t<RANK>
  LocalToGlobal(size_t fragment,
                const distributed_index_t<RANK> &local_index) const {
    distributed_index_t<RANK> global_index{};
    const auto &descriptor = fragments_.at(fragment);
    for (int dim = 0; dim < RANK; ++dim) {
      matx::detail::DistributedCheck(
          local_index[dim] >= 0 && local_index[dim] < descriptor.shape[dim],
          matxInvalidSize, "Local fragment index is out of bounds");
      global_index[dim] = descriptor.origin[dim] + local_index[dim];
    }
    return global_index;
  }

  bool Compatible(const block_distribution_t &other) const noexcept {
    return global_shape_ == other.global_shape_ &&
           fragments_ == other.fragments_;
  }

private:
  void Validate() const {
    matx::detail::DistributedCheck(
        !fragments_.empty(), matxInvalidParameter,
        "A block distribution needs at least one fragment");
    for (int dim = 0; dim < RANK; ++dim) {
      matx::detail::DistributedCheck(
          global_shape_[dim] > 0, matxInvalidSize,
          "Distributed tensor extents must be positive");
    }

    index_t covered = 0;
    for (size_t i = 0; i < fragments_.size(); ++i) {
      const auto &fragment = fragments_[i];
      for (int dim = 0; dim < RANK; ++dim) {
        matx::detail::DistributedCheck(
            fragment.origin[dim] >= 0 && fragment.shape[dim] > 0,
            matxInvalidSize, "Invalid distributed fragment box");
        matx::detail::DistributedCheck(
            fragment.origin[dim] + fragment.shape[dim] <= global_shape_[dim],
            matxInvalidSize,
            "Distributed fragment extends beyond the global shape");
      }
      covered += cuda::std::accumulate(
          fragment.shape.begin(), fragment.shape.end(), index_t{1},
          cuda::std::multiplies<index_t>{});

      for (size_t j = i + 1; j < fragments_.size(); ++j) {
        bool overlaps = true;
        for (int dim = 0; dim < RANK; ++dim) {
          const index_t i_end = fragment.origin[dim] + fragment.shape[dim];
          const index_t j_end =
              fragments_[j].origin[dim] + fragments_[j].shape[dim];
          overlaps &=
              fragment.origin[dim] < j_end && fragments_[j].origin[dim] < i_end;
        }
        matx::detail::DistributedCheck(!overlaps, matxInvalidParameter,
                                       "Distributed block fragments overlap");
      }
    }

    matx::detail::DistributedCheck(
        covered == cuda::std::accumulate(
                       global_shape_.begin(), global_shape_.end(), index_t{1},
                       cuda::std::multiplies<index_t>{}),
        matxInvalidSize,
        "Distributed block fragments do not cover the global tensor");
  }

  distributed_index_t<RANK> global_shape_{};
  std::vector<block_fragment_descriptor_t<RANK>> fragments_;
};

/** A full logical tensor replica on each endpoint. */
template <int RANK> class replicated_distribution_t {
public:
  static_assert(RANK > 0, "Distributed rank-zero tensors are not supported");
  static constexpr int Rank() noexcept { return RANK; }

  replicated_distribution_t(distributed_index_t<RANK> global_shape,
                            std::vector<distributed_endpoint_t> endpoints)
      : global_shape_{global_shape}, endpoints_{std::move(endpoints)} {
    matx::detail::DistributedCheck(
        !endpoints_.empty(), matxInvalidParameter,
        "A replicated distribution needs at least one endpoint");
    for (int dim = 0; dim < RANK; ++dim) {
      matx::detail::DistributedCheck(
          global_shape_[dim] > 0, matxInvalidSize,
          "Distributed tensor extents must be positive");
    }
    for (size_t i = 0; i < endpoints_.size(); ++i) {
      for (size_t j = i + 1; j < endpoints_.size(); ++j) {
        matx::detail::DistributedCheck(endpoints_[i] != endpoints_[j],
                                       matxInvalidParameter,
                                       "Replicated endpoints must be unique");
      }
    }
  }

  const distributed_index_t<RANK> &GlobalShape() const noexcept {
    return global_shape_;
  }
  size_t FragmentCount() const noexcept { return endpoints_.size(); }
  const distributed_endpoint_t &FragmentEndpoint(size_t fragment) const {
    return endpoints_.at(fragment);
  }
  const distributed_index_t<RANK> &LocalShape(size_t) const noexcept {
    return global_shape_;
  }
  distributed_index_t<RANK>
  LocalToGlobal(size_t, const distributed_index_t<RANK> &local_index) const {
    return local_index;
  }
  bool Compatible(const replicated_distribution_t &other) const noexcept {
    return global_shape_ == other.global_shape_ &&
           endpoints_ == other.endpoints_;
  }

private:
  distributed_index_t<RANK> global_shape_{};
  std::vector<distributed_endpoint_t> endpoints_;
};

/**
 * Two-dimensional block-cyclic mapping compatible with the descriptor model
 * used by cuBLASMp and cuSOLVERMp.  Local tensors use packed row-major indices;
 * backend adapters may reinterpret or copy them to a library-specific layout.
 */
class block_cyclic_distribution_t {
public:
  static constexpr int Rank() noexcept { return 2; }

  block_cyclic_distribution_t(distributed_index_t<2> global_shape,
                              distributed_index_t<2> block_shape,
                              distributed_index_t<2> process_grid,
                              std::vector<distributed_endpoint_t> endpoints)
      : global_shape_{global_shape}, block_shape_{block_shape},
        process_grid_{process_grid}, endpoints_{std::move(endpoints)} {
    for (int dim = 0; dim < 2; ++dim) {
      matx::detail::DistributedCheck(
          global_shape_[dim] > 0 && block_shape_[dim] > 0 &&
              process_grid_[dim] > 0,
          matxInvalidSize,
          "Block-cyclic shapes and process-grid extents must be positive");
    }
    matx::detail::DistributedCheck(
        static_cast<index_t>(endpoints_.size()) ==
            process_grid_[0] * process_grid_[1],
        matxInvalidSize,
        "Block-cyclic endpoint count must match the process grid");
  }

  const distributed_index_t<2> &GlobalShape() const noexcept {
    return global_shape_;
  }
  const distributed_index_t<2> &BlockShape() const noexcept {
    return block_shape_;
  }
  const distributed_index_t<2> &ProcessGrid() const noexcept {
    return process_grid_;
  }
  size_t FragmentCount() const noexcept { return endpoints_.size(); }
  const distributed_endpoint_t &FragmentEndpoint(size_t fragment) const {
    return endpoints_.at(fragment);
  }

  distributed_index_t<2> LocalShape(size_t fragment) const {
    const index_t process_row =
        static_cast<index_t>(fragment) / process_grid_[1];
    const index_t process_col =
        static_cast<index_t>(fragment) % process_grid_[1];
    return {OwnedExtent(global_shape_[0], block_shape_[0], process_grid_[0],
                        process_row),
            OwnedExtent(global_shape_[1], block_shape_[1], process_grid_[1],
                        process_col)};
  }

  distributed_index_t<2>
  LocalToGlobal(size_t fragment,
                const distributed_index_t<2> &local_index) const {
    [[maybe_unused]] const auto local_shape = LocalShape(fragment);
    matx::detail::DistributedCheck(
        local_index[0] >= 0 && local_index[0] < local_shape[0] &&
            local_index[1] >= 0 && local_index[1] < local_shape[1],
        matxInvalidSize, "Local block-cyclic index is out of bounds");
    const index_t process_row =
        static_cast<index_t>(fragment) / process_grid_[1];
    const index_t process_col =
        static_cast<index_t>(fragment) % process_grid_[1];
    return {MapIndex(local_index[0], block_shape_[0], process_grid_[0],
                     process_row),
            MapIndex(local_index[1], block_shape_[1], process_grid_[1],
                     process_col)};
  }

  bool Compatible(const block_cyclic_distribution_t &other) const noexcept {
    return global_shape_ == other.global_shape_ &&
           block_shape_ == other.block_shape_ &&
           process_grid_ == other.process_grid_ &&
           endpoints_ == other.endpoints_;
  }

private:
  static index_t OwnedExtent(index_t global, index_t block, index_t processes,
                             index_t coordinate) {
    index_t owned = 0;
    const index_t block_count = (global + block - 1) / block;
    for (index_t global_block = coordinate; global_block < block_count;
         global_block += processes) {
      const index_t begin = global_block * block;
      owned += std::min(block, global - begin);
    }
    return owned;
  }

  static index_t MapIndex(index_t local, index_t block, index_t processes,
                          index_t coordinate) {
    const index_t local_block = local / block;
    const index_t within_block = local % block;
    return (local_block * processes + coordinate) * block + within_block;
  }

  distributed_index_t<2> global_shape_{};
  distributed_index_t<2> block_shape_{};
  distributed_index_t<2> process_grid_{};
  std::vector<distributed_endpoint_t> endpoints_;
};

/** One locally addressable, homogeneous piece of a distributed tensor. */
template <typename T, int RANK> struct local_fragment_t {
  using value_type = T;
  tensor_t<T, RANK> view;
  size_t distribution_index = 0;
  distributed_endpoint_t endpoint;
  std::shared_ptr<void> backend_owner{};
};

/**
 * Binds caller-owned storage to one locally addressable distribution fragment.
 *
 * The pointer is non-owning unless an optional lifetime token is supplied.
 */
template <typename T> struct distributed_local_pointer_t {
  size_t distribution_index = 0;
  T *data = nullptr;
  std::shared_ptr<void> owner{};
};

template <typename Out, typename Op> class distributed_set_op {
public:
  distributed_set_op(Out &out, const Op &op) : out_{out}, op_{op} {}

  template <typename Executor> void run(Executor &&executor) {
    static_assert(is_distributed_executor_v<Executor>,
                  "Distributed assignment requires distributedCUDAExecutor");
    op_.ExecuteTo(out_, executor);
  }

private:
  Out &out_;
  Op op_;
};

template <typename T, int RANK,
          typename Distribution = block_distribution_t<RANK>>
class distributed_tensor_t {
public:
  using distributed_tensor = bool;
  using value_type = T;
  using distribution_type = Distribution;
  using fragment_type = local_fragment_t<T, RANK>;

  static_assert(Distribution::Rank() == RANK,
                "Distribution rank must match distributed tensor rank");

  distributed_tensor_t(Distribution distribution,
                       const distributed_context &context,
                       std::vector<fragment_type> local_fragments)
      : distribution_{std::move(distribution)}, context_id_{context.Id()},
        process_rank_{context.ProcessRank()},
        local_fragments_{std::move(local_fragments)} {
    Validate(context);
  }

  distributed_tensor_t(const distributed_tensor_t &) = default;
  distributed_tensor_t(distributed_tensor_t &&) noexcept = default;

  static constexpr int Rank() noexcept { return RANK; }
  index_t Size(int dim) const noexcept {
    return distribution_.GlobalShape()[dim];
  }
  const distributed_index_t<RANK> &Shape() const noexcept {
    return distribution_.GlobalShape();
  }
  index_t TotalSize() const noexcept {
    const auto &shape = distribution_.GlobalShape();
    return cuda::std::accumulate(shape.begin(), shape.end(), index_t{1},
                                 cuda::std::multiplies<index_t>{});
  }
  uint64_t ContextId() const noexcept { return context_id_; }
  int ProcessRank() const noexcept { return process_rank_; }
  const Distribution &DistributionDescriptor() const noexcept {
    return distribution_;
  }

  size_t LocalFragmentCount() const noexcept { return local_fragments_.size(); }
  tensor_t<T, RANK> &LocalView(size_t local_fragment) {
    return local_fragments_.at(local_fragment).view;
  }
  const tensor_t<T, RANK> &LocalView(size_t local_fragment) const {
    return local_fragments_.at(local_fragment).view;
  }
  const fragment_type &LocalFragment(size_t local_fragment) const {
    return local_fragments_.at(local_fragment);
  }

  const fragment_type &LocalFragmentForDistributionIndex(size_t index) const {
    for (const auto &fragment : local_fragments_) {
      if (fragment.distribution_index == index) {
        return fragment;
      }
    }
    MATX_THROW(matxInvalidParameter,
               "Distribution fragment is not local to this process");
  }

  fragment_type &LocalFragmentForDistributionIndex(size_t index) {
    for (auto &fragment : local_fragments_) {
      if (fragment.distribution_index == index) {
        return fragment;
      }
    }
    MATX_THROW(matxInvalidParameter,
               "Distribution fragment is not local to this process");
  }

  const std::string str() const { return "distributed_tensor"; }

  template <typename Op>
    requires(is_distributed_expression_v<Op>)
  [[nodiscard]] auto operator=(const Op &op) {
    return distributed_set_op<distributed_tensor_t, Op>{*this, op};
  }

  [[nodiscard]] auto operator=(const distributed_tensor_t &other) {
    return distributed_set_op<distributed_tensor_t, distributed_tensor_t>{
        *this, other};
  }

  template <typename Out, typename Executor>
  void MaterializeTo(Out &out, Executor &executor) const {
    static_assert(is_distributed_executor_v<Executor>,
                  "Distributed materialization requires distributedCUDAExecutor");
    static_assert(Out::Rank() == RANK,
                  "Regular and distributed tensor ranks must match");
    static_assert(
        std::is_same_v<remove_cvref_t<typename Out::value_type>,
                       remove_cvref_t<T>>,
        "Distributed-to-regular assignment requires identical element types");

    matx::detail::DistributedCheck(
        executor.ContextId() == context_id_, matxInvalidExecutor,
        "Distributed tensor and executor contexts do not match");
    matx::detail::DistributedCheck(
        executor.Context().ProcessCount() == 1, matxNotSupported,
        "Multi-process materialization needs a collective backend");
    for (int dim = 0; dim < RANK; ++dim) {
      matx::detail::DistributedCheck(
          out.Size(dim) == Size(dim), matxInvalidSize,
          "Regular destination shape must equal the global shape");
    }

    RejectAliasedDestination(out);
    const auto out_location = PointerLocation(out.Data());
    bool copied_replica = false;
    for (const auto &fragment : local_fragments_) {
      if constexpr (std::is_same_v<Distribution,
                                   replicated_distribution_t<RANK>>) {
        if (copied_replica) {
          continue;
        }
        copied_replica = true;
      }

      const auto local_shape =
          distribution_.LocalShape(fragment.distribution_index);
      const index_t local_size = cuda::std::accumulate(
          local_shape.begin(), local_shape.end(), index_t{1},
          cuda::std::multiplies<index_t>{});
      if (local_size == 0) {
        continue;
      }
      const auto source_location = PointerLocation(fragment.view.Data());

      executor.ForEndpoint(
          fragment.endpoint, [&](cudaExecutor &local_executor) {
            EnableDirectAccess(fragment.endpoint.device_id, source_location,
                               out_location);

            distributed_index_t<RANK> local_zero{};
            const auto global_zero = distribution_.LocalToGlobal(
                fragment.distribution_index, local_zero);
            const index_t global_begin = detail::DistributedFlatten<RANK>(
                global_zero, distribution_.GlobalShape());
            const auto local_last =
                detail::DistributedUnflatten<RANK>(local_size - 1, local_shape);
            const auto global_last = distribution_.LocalToGlobal(
                fragment.distribution_index, local_last);
            const index_t global_end = detail::DistributedFlatten<RANK>(
                global_last, distribution_.GlobalShape());

            if (fragment.view.IsContiguous() && out.IsContiguous() &&
                global_end - global_begin + 1 == local_size) {
              DirectCopy(out.Data() + global_begin, out_location,
                         fragment.view.Data(), source_location,
                         static_cast<size_t>(local_size) * sizeof(T),
                         local_executor.getStream());
              return;
            }

            T *run_source = nullptr;
            T *run_destination = nullptr;
            T *previous_source = nullptr;
            T *previous_destination = nullptr;
            size_t run_elements = 0;
            auto flush = [&]() {
              if (run_elements == 0) {
                return;
              }
              DirectCopy(run_destination, out_location, run_source,
                         source_location, run_elements * sizeof(T),
                         local_executor.getStream());
              run_elements = 0;
            };

            for (index_t linear = 0; linear < local_size; ++linear) {
              const auto local_index =
                  detail::DistributedUnflatten<RANK>(linear, local_shape);
              const auto global_index = distribution_.LocalToGlobal(
                  fragment.distribution_index, local_index);
              index_t source_offset = 0;
              index_t destination_offset = 0;
              for (int dim = 0; dim < RANK; ++dim) {
                source_offset += local_index[dim] * fragment.view.Stride(dim);
                destination_offset += global_index[dim] * out.Stride(dim);
              }
              T *source = fragment.view.Data() + source_offset;
              T *destination = out.Data() + destination_offset;
              if (run_elements != 0 &&
                  (source != previous_source + 1 ||
                   destination != previous_destination + 1)) {
                flush();
              }
              if (run_elements == 0) {
                run_source = source;
                run_destination = destination;
              }
              previous_source = source;
              previous_destination = destination;
              ++run_elements;
            }
            flush();
          });
      executor.RecordTransfer();
    }
  }

  template <typename Out, typename Executor>
  void ExecuteTo(Out &out, Executor &executor) const {
    static_assert(
        is_distributed_tensor_v<Out>,
        "Distributed copies require a distributed tensor destination");
    static_assert(std::is_same_v<typename Out::value_type, T>,
                  "Distributed copies require identical element types");
    ValidateCompatibleOutput(out, executor);

    for (const auto &out_fragment : out.local_fragments_) {
      const auto &in_fragment =
          LocalFragmentForDistributionIndex(out_fragment.distribution_index);
      executor.ForEndpoint(out_fragment.endpoint,
                           [&](cudaExecutor &local_executor) {
                             (out.LocalFragmentForDistributionIndex(
                                     out_fragment.distribution_index)
                                  .view = in_fragment.view)
                                 .run(local_executor);
                           });
    }
  }

  template <typename U, int R, typename D> friend class distributed_tensor_t;

private:
  struct pointer_location_t {
    cudaMemoryType type = cudaMemoryTypeUnregistered;
    int device = -1;
  };

  static pointer_location_t PointerLocation(const void *pointer) {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status == cudaErrorInvalidValue) {
      (void)cudaGetLastError();
      return {};
    }
    MATX_CUDA_CHECK(status);
    return {attributes.type, attributes.device};
  }

  static void EnableDirectAccess(int source_endpoint_device,
                                 const pointer_location_t &source,
                                 const pointer_location_t &destination) {
    if (source.type != cudaMemoryTypeDevice ||
        destination.type != cudaMemoryTypeDevice ||
        source.device == destination.device) {
      return;
    }

    matx::detail::DistributedCheck(
        source.device == source_endpoint_device, matxInvalidParameter,
        "Distributed fragment pointer belongs to a different CUDA device");
    int can_access = 0;
    MATX_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, source.device,
                                            destination.device));
    matx::detail::DistributedCheck(
        can_access != 0, matxNotSupported,
        "Distributed materialization requires direct peer or unified memory "
        "access; host staging is disabled");

    const cudaError_t status =
        cudaDeviceEnablePeerAccess(destination.device, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
      (void)cudaGetLastError();
    } else {
      MATX_CUDA_CHECK(status);
    }
  }

  static void DirectCopy(T *destination,
                         const pointer_location_t &destination_location,
                         const T *source,
                         const pointer_location_t &source_location,
                         size_t bytes, cudaStream_t stream) {
    if (source_location.type == cudaMemoryTypeDevice &&
        destination_location.type == cudaMemoryTypeDevice &&
        source_location.device != destination_location.device) {
      MATX_CUDA_CHECK(
          cudaMemcpyPeerAsync(destination, destination_location.device, source,
                              source_location.device, bytes, stream));
    } else {
      MATX_CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes,
                                      cudaMemcpyDefault, stream));
    }
  }

  template <typename Other, typename Executor>
  void ValidateCompatibleOutput(const Other &other,
                                const Executor &executor) const {
    static_assert(Other::Rank() == RANK,
                  "Distributed operands must have the same rank");
    static_assert(
        std::is_same_v<typename Other::distribution_type, Distribution>,
        "Distributed operands must use the same distribution type");
    matx::detail::DistributedCheck(
        context_id_ == other.ContextId() && context_id_ == executor.ContextId(),
        matxInvalidExecutor,
        "Distributed operands and executor contexts do not match");
    matx::detail::DistributedCheck(
        distribution_.Compatible(other.DistributionDescriptor()),
        matxInvalidParameter,
        "Distributed operands must have identical fragment layouts");
  }

  void Validate(const distributed_context &context) const {
    std::vector<bool> found(distribution_.FragmentCount(), false);
    for (const auto &fragment : local_fragments_) {
      matx::detail::DistributedCheck(
          fragment.distribution_index < distribution_.FragmentCount(),
          matxInvalidParameter,
          "Local fragment has an invalid distribution index");
      matx::detail::DistributedCheck(
          !found[fragment.distribution_index], matxInvalidParameter,
          "Local distribution fragment appears more than once");
      found[fragment.distribution_index] = true;

      const auto &expected_endpoint =
          distribution_.FragmentEndpoint(fragment.distribution_index);
      matx::detail::DistributedCheck(
          fragment.endpoint == expected_endpoint, matxInvalidParameter,
          "Local fragment endpoint does not match its distribution");
      matx::detail::DistributedCheck(
          fragment.endpoint.process_rank == context.ProcessRank() &&
              context.IsLocalDevice(fragment.endpoint.device_id),
          matxInvalidParameter,
          "Local fragment is not owned by this distributed context");
      const auto expected_shape =
          distribution_.LocalShape(fragment.distribution_index);
      for (int dim = 0; dim < RANK; ++dim) {
        matx::detail::DistributedCheck(
            fragment.view.Size(dim) == expected_shape[dim], matxInvalidSize,
            "Local tensor shape does not match its distribution fragment");
      }
    }

    for (size_t i = 0; i < distribution_.FragmentCount(); ++i) {
      if (distribution_.FragmentEndpoint(i).process_rank ==
          context.ProcessRank()) {
        matx::detail::DistributedCheck(
            found[i], matxInvalidParameter,
            "A locally-owned distribution fragment has no local tensor");
      } else {
        matx::detail::DistributedCheck(
            !found[i], matxInvalidParameter,
            "Remote fragments cannot store local pointers");
      }
    }
  }

  template <typename Out> void RejectAliasedDestination(const Out &out) const {
    const auto [out_begin, out_end] = AddressRange(out);

    for (const auto &fragment : local_fragments_) {
      const auto [in_begin, in_end] = AddressRange(fragment.view);
      matx::detail::DistributedCheck(
          out_end <= in_begin || in_end <= out_begin, matxInvalidParameter,
          "Regular destination aliases a distributed source fragment");
    }
  }

  template <typename View>
  static std::pair<uintptr_t, uintptr_t> AddressRange(const View &view) {
    index_t first_offset = 0;
    index_t last_offset = 0;
    for (int dim = 0; dim < RANK; ++dim) {
      const index_t extent = (view.Size(dim) - 1) * view.Stride(dim);
      first_offset += std::min<index_t>(0, extent);
      last_offset += std::max<index_t>(0, extent);
    }
    const auto begin = reinterpret_cast<uintptr_t>(view.Data() + first_offset);
    const auto end = reinterpret_cast<uintptr_t>(view.Data() + last_offset + 1);
    return {begin, end};
  }

  Distribution distribution_;
  uint64_t context_id_;
  int process_rank_;
  std::vector<fragment_type> local_fragments_;
};

template <typename Function, typename... Inputs> class distributed_apply_op {
private:
  using first_input_type =
      remove_cvref_t<std::tuple_element_t<0, std::tuple<Inputs...>>>;

public:
  using distributed_expression = bool;
  using value_type = remove_cvref_t<
      std::invoke_result_t<Function, typename Inputs::value_type...>>;
  using distribution_type = typename first_input_type::distribution_type;

  static_assert(sizeof...(Inputs) > 0,
                "Distributed apply requires at least one input");
  static_assert((is_distributed_tensor_v<Inputs> && ...),
                "Apply inputs must all be distributed tensors");
  static_assert(((remove_cvref_t<Inputs>::Rank() == first_input_type::Rank()) &&
                 ...),
                "Distributed apply inputs must have the same rank");
  static_assert(
      (std::is_same_v<typename remove_cvref_t<Inputs>::distribution_type,
                      distribution_type> &&
       ...),
      "Distributed apply inputs must use the same distribution type");

  distributed_apply_op(Function function, const Inputs &...inputs)
      : function_{std::move(function)}, inputs_{inputs...} {
    ValidateInputs();
  }

  static constexpr int Rank() noexcept { return first_input_type::Rank(); }
  index_t Size(int dim) const noexcept { return FirstInput().Size(dim); }

  template <typename Out, typename Executor>
  void ExecuteTo(Out &out, Executor &executor) const {
    static_assert(is_distributed_tensor_v<Out>,
                  "Distributed apply requires a distributed tensor output");
    static_assert(Out::Rank() == Rank(),
                  "Distributed apply output rank must match its inputs");
    static_assert(
        std::is_same_v<typename Out::value_type, value_type>,
        "Distributed apply result and output types must match exactly");
    static_assert(
        std::is_same_v<typename Out::distribution_type, distribution_type>,
        "Distributed apply output distribution type must match its inputs");

    matx::detail::DistributedCheck(
        out.ContextId() == FirstInput().ContextId() &&
            out.ContextId() == executor.ContextId(),
        matxInvalidExecutor,
        "Distributed apply operands and executor contexts do not match");
    matx::detail::DistributedCheck(
        out.DistributionDescriptor().Compatible(
            FirstInput().DistributionDescriptor()),
        matxInvalidParameter,
        "Distributed apply output layout must match its inputs");

    for (size_t local = 0; local < out.LocalFragmentCount(); ++local) {
      auto &out_fragment = out.LocalFragment(local);
      executor.ForEndpoint(
          out_fragment.endpoint, [&](cudaExecutor &local_executor) {
            auto local_op = MakeLocalOp(out_fragment.distribution_index);
            static_assert(
                std::is_same_v<
                    remove_cvref_t<typename decltype(local_op)::value_type>,
                    remove_cvref_t<typename Out::value_type>>,
                "Distributed apply callable result must exactly match the "
                "output type");
            (out.LocalView(local) = local_op).run(local_executor);
          });
    }
  }

private:
  const first_input_type &FirstInput() const { return std::get<0>(inputs_); }

  void ValidateInputs() const {
    const auto &first = FirstInput();
    std::apply([&](const auto &...input) { (ValidateOne(first, input), ...); },
               inputs_);
  }

  template <typename Input>
  static void ValidateOne(const first_input_type &first, const Input &input) {
    matx::detail::DistributedCheck(
        first.ContextId() == input.ContextId(), matxInvalidExecutor,
        "Distributed apply inputs use different contexts");
    matx::detail::DistributedCheck(
        first.DistributionDescriptor().Compatible(
            input.DistributionDescriptor()),
        matxInvalidParameter,
        "Distributed apply inputs must have identical fragment layouts");
  }

  auto MakeLocalOp(size_t distribution_index) const {
    return std::apply(
        [&](const auto &...input) {
          return matx::apply(
              function_,
              input.LocalFragmentForDistributionIndex(distribution_index)
                  .view...);
        },
        inputs_);
  }

  Function function_;
  std::tuple<Inputs...> inputs_;
};

namespace detail {

template <typename Function, typename... Inputs>
auto MakeDistributedApply(Function &&function, const Inputs &...inputs) {
  return distributed_apply_op<
      remove_cvref_t<Function>, remove_cvref_t<Inputs>...>{
      std::forward<Function>(function), inputs...};
}

} // namespace detail

/**
 * Executes an existing MatX transform independently on aligned local fragments.
 *
 * The leading dimensions may be partitioned, while the trailing operation
 * dimensions must be fully local. No communication or redistribution occurs.
 */
namespace detail {

template <typename ValueType, int OPERATION_DIMS, typename Factory,
          typename... Inputs>
class distributed_local_transform_op {
private:
  using first_input_type =
      remove_cvref_t<std::tuple_element_t<0, std::tuple<Inputs...>>>;

public:
  using distributed_expression = bool;
  using value_type = ValueType;

  static_assert(sizeof...(Inputs) > 0,
                "A distributed local transform needs at least one input");
  static_assert((is_distributed_tensor_v<Inputs> && ...),
                "Distributed local transform inputs must be distributed tensors");
  static_assert(((remove_cvref_t<Inputs>::Rank() ==
                  first_input_type::Rank()) &&
                 ...),
                "First-pass distributed local transforms require equal input ranks");
  static_assert(OPERATION_DIMS > 0 &&
                    OPERATION_DIMS <= first_input_type::Rank(),
                "Invalid number of local transform dimensions");

  distributed_local_transform_op(Factory factory, const Inputs &...inputs)
      : factory_{std::move(factory)}, inputs_{inputs...} {
    ValidateInputContexts();
  }

  template <typename Out, typename Executor>
  void ExecuteTo(Out &out, Executor &executor) const {
    static_assert(is_distributed_tensor_v<Out>,
                  "A distributed local transform needs a distributed output");
    static_assert(is_distributed_executor_v<Executor>,
                  "A distributed local transform needs distributedCUDAExecutor");
    static_assert(Out::Rank() == first_input_type::Rank(),
                  "First-pass distributed local transforms preserve rank");
    static_assert(
        std::is_same_v<remove_cvref_t<typename Out::value_type>,
                       remove_cvref_t<ValueType>>,
        "Distributed local transform output type does not match");

    matx::detail::DistributedCheck(
        out.ContextId() == FirstInput().ContextId() &&
            out.ContextId() == executor.ContextId(),
        matxInvalidExecutor,
        "Distributed local transform operands and executor contexts do not match");
    std::apply(
        [&](const auto &...input) { (ValidateAligned(out, input), ...); },
        inputs_);

    for (size_t local = 0; local < out.LocalFragmentCount(); ++local) {
      const auto &out_fragment = out.LocalFragment(local);
      executor.ForEndpoint(
          out_fragment.endpoint, [&](cudaExecutor &local_executor) {
            auto local_op = MakeLocalOp(out_fragment.distribution_index);
            (out.LocalView(local) = local_op).run(local_executor);
          });
    }
  }

private:
  const first_input_type &FirstInput() const { return std::get<0>(inputs_); }

  void ValidateInputContexts() const {
    const uint64_t context_id = FirstInput().ContextId();
    std::apply(
        [&](const auto &...input) {
          (matx::detail::DistributedCheck(
               input.ContextId() == context_id, matxInvalidExecutor,
               "Distributed local transform inputs use different contexts"),
           ...);
        },
        inputs_);
  }

  template <typename Distribution>
  static void ValidateOperationDimensionsLocal(
      const Distribution &distribution) {
    constexpr int rank = Distribution::Rank();
    for (size_t fragment = 0; fragment < distribution.FragmentCount();
         ++fragment) {
      distributed_index_t<rank> local_zero{};
      const auto origin =
          distribution.LocalToGlobal(fragment, local_zero);
      const auto local_shape = distribution.LocalShape(fragment);
      for (int dim = rank - OPERATION_DIMS; dim < rank; ++dim) {
        matx::detail::DistributedCheck(
            origin[dim] == 0 &&
                local_shape[dim] == distribution.GlobalShape()[dim],
            matxInvalidParameter,
            "Distributed local transform dimensions cannot be partitioned");
      }
    }
  }

  template <typename Out, typename Input>
  static void ValidateAligned(const Out &out, const Input &input) {
    constexpr int rank = Out::Rank();
    const auto &out_distribution = out.DistributionDescriptor();
    const auto &input_distribution = input.DistributionDescriptor();
    matx::detail::DistributedCheck(
        out_distribution.FragmentCount() ==
            input_distribution.FragmentCount(),
        matxInvalidParameter,
        "Distributed local transform fragment counts do not match");

    for (int dim = 0; dim < rank - OPERATION_DIMS; ++dim) {
      matx::detail::DistributedCheck(
          out_distribution.GlobalShape()[dim] ==
              input_distribution.GlobalShape()[dim],
          matxInvalidSize,
          "Distributed local transform batch dimensions do not match");
    }

    for (size_t fragment = 0;
         fragment < out_distribution.FragmentCount(); ++fragment) {
      matx::detail::DistributedCheck(
          out_distribution.FragmentEndpoint(fragment) ==
              input_distribution.FragmentEndpoint(fragment),
          matxInvalidParameter,
          "Distributed local transform endpoints do not match");
      distributed_index_t<rank> local_zero{};
      const auto out_origin =
          out_distribution.LocalToGlobal(fragment, local_zero);
      const auto input_origin =
          input_distribution.LocalToGlobal(fragment, local_zero);
      const auto out_shape = out_distribution.LocalShape(fragment);
      const auto input_shape = input_distribution.LocalShape(fragment);
      for (int dim = 0; dim < rank - OPERATION_DIMS; ++dim) {
        matx::detail::DistributedCheck(
            out_origin[dim] == input_origin[dim] &&
                out_shape[dim] == input_shape[dim],
            matxInvalidParameter,
            "Distributed local transform batch partitions do not match");
      }
    }

    ValidateOperationDimensionsLocal(out_distribution);
    ValidateOperationDimensionsLocal(input_distribution);
  }

  auto MakeLocalOp(size_t distribution_index) const {
    return std::apply(
        [&](const auto &...input) {
          return factory_(
              input.LocalFragmentForDistributionIndex(distribution_index)
                  .view...);
        },
        inputs_);
  }

  Factory factory_;
  std::tuple<Inputs...> inputs_;
};

template <typename ValueType, int OPERATION_DIMS, typename Factory,
          typename... Inputs>
auto make_distributed_local_transform(Factory &&factory,
                                      const Inputs &...inputs) {
  return distributed_local_transform_op<
      ValueType, OPERATION_DIMS, remove_cvref_t<Factory>,
      remove_cvref_t<Inputs>...>{std::forward<Factory>(factory), inputs...};
}

} // namespace detail

template <typename T, typename Distribution>
auto make_distributed_tensor(
    const Distribution &distribution, const distributed_context &context,
    matxMemorySpace_t memory_space = MATX_DEVICE_MEMORY) {
  constexpr int RANK = Distribution::Rank();
  std::vector<local_fragment_t<T, RANK>> fragments;
  for (size_t i = 0; i < distribution.FragmentCount(); ++i) {
    const auto &endpoint = distribution.FragmentEndpoint(i);
    if (endpoint.process_rank != context.ProcessRank()) {
      continue;
    }
    matx::detail::DistributedCheck(
        context.IsLocalDevice(endpoint.device_id), matxInvalidParameter,
        "Distribution references a device outside the context");
    matx::detail::distributed_device_guard guard{endpoint.device_id};
    auto local = make_tensor<T>(distribution.LocalShape(i), memory_space);
    fragments.push_back({std::move(local), i, endpoint, {}});
  }
  return distributed_tensor_t<T, RANK, Distribution>{distribution, context,
                                                     std::move(fragments)};
}

template <typename T, typename Distribution>
auto make_distributed_tensor(
    const Distribution &distribution, const distributed_context &context,
    const std::vector<distributed_local_pointer_t<T>> &local_pointers) {
  constexpr int RANK = Distribution::Rank();
  std::vector<local_fragment_t<T, RANK>> fragments;
  fragments.reserve(local_pointers.size());
  std::vector<bool> found(distribution.FragmentCount(), false);

  for (const auto &binding : local_pointers) {
    matx::detail::DistributedCheck(
        binding.distribution_index < distribution.FragmentCount(),
        matxInvalidParameter,
        "Distributed pointer has an invalid distribution index");
    matx::detail::DistributedCheck(
        !found[binding.distribution_index], matxInvalidParameter,
        "A distributed pointer index may appear only once");
    matx::detail::DistributedCheck(
        binding.data != nullptr, matxInvalidParameter,
        "Distributed local pointers cannot be null");

    const auto &endpoint =
        distribution.FragmentEndpoint(binding.distribution_index);
    matx::detail::DistributedCheck(
        endpoint.process_rank == context.ProcessRank() &&
            context.IsLocalDevice(endpoint.device_id),
        matxInvalidParameter,
        "Distributed pointers may bind only locally owned fragments");
    found[binding.distribution_index] = true;

    matx::detail::distributed_device_guard guard{endpoint.device_id};
    auto local = make_tensor<T>(
        binding.data, distribution.LocalShape(binding.distribution_index),
        false);
    fragments.push_back({std::move(local), binding.distribution_index,
                         endpoint, binding.owner});
  }

  return distributed_tensor_t<T, RANK, Distribution>{
      distribution, context, std::move(fragments)};
}

template <typename T, typename Distribution>
auto make_distributed_tensor(const Distribution &distribution,
                             const distributed_context &context,
                             const std::vector<T *> &local_pointers) {
  std::vector<distributed_local_pointer_t<T>> bindings;
  bindings.reserve(local_pointers.size());
  size_t local_pointer = 0;
  for (size_t fragment = 0; fragment < distribution.FragmentCount();
       ++fragment) {
    if (distribution.FragmentEndpoint(fragment).process_rank !=
        context.ProcessRank()) {
      continue;
    }
    matx::detail::DistributedCheck(
        local_pointer < local_pointers.size(), matxInvalidParameter,
        "Too few pointers for locally owned distribution fragments");
    bindings.push_back(
        {fragment, local_pointers[local_pointer], {}});
    ++local_pointer;
  }
  matx::detail::DistributedCheck(
      local_pointer == local_pointers.size(), matxInvalidParameter,
      "Too many pointers for locally owned distribution fragments");
  return make_distributed_tensor(distribution, context, bindings);
}

template <typename T, typename Distribution>
auto make_distributed_tensor(
    const Distribution &distribution, const distributed_context &context,
    T *local_pointer, std::shared_ptr<void> owner = {}) {
  size_t local_fragment_count = 0;
  size_t local_distribution_index = 0;
  for (size_t fragment = 0; fragment < distribution.FragmentCount();
       ++fragment) {
    if (distribution.FragmentEndpoint(fragment).process_rank ==
        context.ProcessRank()) {
      local_distribution_index = fragment;
      ++local_fragment_count;
    }
  }
  matx::detail::DistributedCheck(
      local_fragment_count == 1, matxInvalidParameter,
      "The single-pointer helper requires exactly one local fragment");
  return make_distributed_tensor(
      distribution, context,
      std::vector<distributed_local_pointer_t<T>>{
          {local_distribution_index, local_pointer, std::move(owner)}});
}

} // namespace experimental
} // namespace matx
