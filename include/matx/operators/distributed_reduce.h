////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (c) 2026, NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "matx/core/distributed_tensor.h"
#include "matx/operators/all.h"
#include "matx/operators/any.h"
#include "matx/operators/max.h"
#include "matx/operators/mean.h"
#include "matx/operators/min.h"
#include "matx/operators/prod.h"
#include "matx/operators/sum.h"

namespace matx::experimental::detail {

enum class distributed_reduction_kind { sum, prod, min, max, mean, all, any };
template <typename> inline constexpr bool dependent_false_v = false;

#ifdef MATX_EN_NCCL
template <typename T> ncclDataType_t DistributedNcclType() {
  using U = matx::detail::convert_matx_type_t<remove_cvref_t<T>>;
  if constexpr (std::is_same_v<U, int8_t>)
    return ncclInt8;
  else if constexpr (std::is_same_v<U, uint8_t>)
    return ncclUint8;
  else if constexpr (std::is_same_v<U, int32_t>)
    return ncclInt32;
  else if constexpr (std::is_same_v<U, uint32_t>)
    return ncclUint32;
  else if constexpr (std::is_same_v<U, int64_t>)
    return ncclInt64;
  else if constexpr (std::is_same_v<U, uint64_t>)
    return ncclUint64;
  else if constexpr (std::is_same_v<U, __half>)
    return ncclFloat16;
  else if constexpr (std::is_same_v<U, float>)
    return ncclFloat32;
  else if constexpr (std::is_same_v<U, double>)
    return ncclFloat64;
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 21000
  else if constexpr (std::is_same_v<U, __nv_bfloat16>)
    return ncclBfloat16;
#endif
  else
    static_assert(dependent_false_v<U>,
                  "Distributed reduction type is not supported by NCCL");
}

template <typename T> struct nccl_storage_type { using type = T; };
template <typename T> struct nccl_storage_type<cuda::std::complex<T>> {
  using type = T;
};
#endif

#ifdef __CUDACC__
template <distributed_reduction_kind KIND, typename T>
__device__ T DistributedCombine(const T &a, const T &b) {
  if constexpr (KIND == distributed_reduction_kind::sum ||
                KIND == distributed_reduction_kind::mean)
    return a + b;
  else if constexpr (KIND == distributed_reduction_kind::prod)
    return a * b;
  else if constexpr (KIND == distributed_reduction_kind::min)
    return b < a ? b : a;
  else if constexpr (KIND == distributed_reduction_kind::max)
    return b > a ? b : a;
  else if constexpr (KIND == distributed_reduction_kind::all)
    return static_cast<T>((a != T{}) && (b != T{}));
  else
    return static_cast<T>((a != T{}) || (b != T{}));
}

template <distributed_reduction_kind KIND, typename T>
__global__ void DistributedInitializeKernel(T *data, index_t count) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  if constexpr (KIND == distributed_reduction_kind::prod ||
                KIND == distributed_reduction_kind::all)
    data[i] = T{1};
  else if constexpr (KIND == distributed_reduction_kind::min)
    data[i] = matx::detail::maxVal<T>();
  else if constexpr (KIND == distributed_reduction_kind::max)
    data[i] = matx::detail::minVal<T>();
  else
    data[i] = T{};
}

template <distributed_reduction_kind KIND, typename T>
__global__ void DistributedScatterKernel(T *dense, const T *partial,
                                         const index_t *mapping,
                                         index_t count) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const index_t destination = mapping[i];
    dense[destination] =
        DistributedCombine<KIND>(dense[destination], partial[i]);
  }
}

template <typename T>
__global__ void DistributedToFlagsKernel(uint8_t *out, const T *in,
                                         index_t count) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = in[i] != T{} ? uint8_t{1} : uint8_t{0};
}

template <typename T>
__global__ void DistributedFromFlagsKernel(T *out, const uint8_t *in,
                                           index_t count) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = static_cast<T>(in[i]);
}

template <typename T>
__global__ void DistributedMeanScaleKernel(T *data, index_t count,
                                           index_t divisor) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    data[i] /= static_cast<T>(divisor);
}
#endif

template <distributed_reduction_kind KIND, int D, typename Out, typename In,
          typename Executor>
void RunLocalReduction(Out &out, const In &in, Executor &executor,
                       const cuda::std::array<int, D> &axes) {
  int dims[D];
  for (int i = 0; i < D; ++i)
    dims[i] = axes[i];
  if constexpr (KIND == distributed_reduction_kind::sum ||
                KIND == distributed_reduction_kind::mean) {
    if constexpr (D == In::Rank())
      (out = matx::sum(in)).run(executor);
    else
      (out = matx::sum(in, dims)).run(executor);
  } else if constexpr (KIND == distributed_reduction_kind::prod) {
    if constexpr (D == In::Rank())
      (out = matx::prod(in)).run(executor);
    else
      (out = matx::prod(in, dims)).run(executor);
  } else if constexpr (KIND == distributed_reduction_kind::min) {
    if constexpr (D == In::Rank())
      (out = matx::min(in)).run(executor);
    else
      (out = matx::min(in, dims)).run(executor);
  } else if constexpr (KIND == distributed_reduction_kind::max) {
    if constexpr (D == In::Rank())
      (out = matx::max(in)).run(executor);
    else
      (out = matx::max(in, dims)).run(executor);
  } else if constexpr (KIND == distributed_reduction_kind::all) {
    if constexpr (D == In::Rank())
      (out = matx::all(in)).run(executor);
    else
      (out = matx::all(in, dims)).run(executor);
  } else {
    if constexpr (D == In::Rank())
      (out = matx::any(in)).run(executor);
    else
      (out = matx::any(in, dims)).run(executor);
  }
}

template <distributed_reduction_kind KIND, typename Input, int D>
class distributed_reduction_op {
public:
  using distributed_expression = bool;
  using value_type = typename remove_cvref_t<Input>::value_type;
  static constexpr int InputRank = remove_cvref_t<Input>::Rank();
  static constexpr int OutputRank = InputRank - D;
  static_assert(D > 0 && D <= InputRank, "Invalid distributed reduction rank");
  static_assert(
      !((KIND == distributed_reduction_kind::prod ||
         KIND == distributed_reduction_kind::min ||
         KIND == distributed_reduction_kind::max) &&
        is_complex_v<value_type>),
      "Complex distributed reductions support sum, mean, all, and any");

  distributed_reduction_op(const Input &input, const int (&axes)[D])
      : input_{input} {
    for (int i = 0; i < D; ++i)
      axes_[i] = axes[i];
    InitializeShape();
  }

  explicit distributed_reduction_op(const Input &input) : input_{input} {
    static_assert(D == InputRank,
                  "The no-axis reduction must reduce every dimension");
    for (int i = 0; i < D; ++i)
      axes_[i] = i;
    InitializeShape();
  }

  static constexpr int Rank() noexcept { return OutputRank; }
  index_t Size(int dim) const noexcept { return output_shape_[dim]; }

  template <typename Out, typename Executor>
  void ExecuteTo(Out &out, Executor &executor) const {
    static_assert(is_distributed_tensor_v<Out>,
                  "Distributed reductions need a distributed tensor output");
    static_assert(is_distributed_executor_v<Executor>,
                  "Distributed reductions need distributedCUDAExecutor");
    static_assert(Out::Rank() == OutputRank,
                  "Distributed reduction output rank does not match");
    static_assert(std::is_same_v<typename Out::value_type, value_type>,
                  "Distributed reduction input and output types must match");
    static_assert(is_replicated_distribution_v<typename Out::distribution_type>,
                  "Distributed reduction output must be replicated");
#ifndef MATX_EN_NCCL
    MATX_THROW(matxNotSupported,
               "Distributed reductions require MATX_EN_NCCL=ON");
#else
    Validate(out, executor);
    ExecuteNccl(out, executor);
#endif
  }

private:
  void InitializeShape() {
    std::array<bool, InputRank> reduced{};
    for (int i = 0; i < D; ++i) {
      matx::detail::DistributedCheck(
          axes_[i] >= 0 && axes_[i] < InputRank, matxInvalidDim,
          "Distributed reduction axis is out of range");
      matx::detail::DistributedCheck(
          !reduced[axes_[i]], matxInvalidDim,
          "Distributed reduction axes must be unique");
      reduced[axes_[i]] = true;
    }
    int output_dim = 0;
    for (int dim = 0; dim < InputRank; ++dim)
      if (!reduced[dim])
        output_shape_[output_dim++] = input_.Size(dim);
  }

#ifdef MATX_EN_NCCL
  template <typename Out, typename Executor>
  void Validate(const Out &out, const Executor &executor) const {
    matx::detail::DistributedCheck(
        input_.ContextId() == out.ContextId() &&
            input_.ContextId() == executor.ContextId(),
        matxInvalidExecutor,
        "Distributed reduction operands and executor contexts do not match");
    matx::detail::DistributedCheck(
        executor.HasNcclTopology(), matxInvalidExecutor,
        "Distributed reduction executor has no NCCL topology");
    for (int dim = 0; dim < OutputRank; ++dim)
      matx::detail::DistributedCheck(
          out.Size(dim) == output_shape_[dim], matxInvalidSize,
          "Distributed reduction output shape is incorrect");
    const auto &endpoints = executor.CollectiveEndpoints();
    const auto &distribution = out.DistributionDescriptor();
    matx::detail::DistributedCheck(
        distribution.FragmentCount() == endpoints.size(), matxInvalidParameter,
        "Reduction output must replicate to every NCCL endpoint");
    for (size_t i = 0; i < endpoints.size(); ++i)
      matx::detail::DistributedCheck(
          distribution.FragmentEndpoint(i) == endpoints[i],
          matxInvalidParameter,
          "Reduction output endpoint order must match NCCL rank order");
    const auto &input_distribution = input_.DistributionDescriptor();
    for (size_t i = 0; i < input_distribution.FragmentCount(); ++i) {
      const auto endpoint = input_distribution.FragmentEndpoint(i);
      matx::detail::DistributedCheck(
          std::find(endpoints.begin(), endpoints.end(), endpoint) !=
              endpoints.end(),
          matxInvalidParameter,
          "Reduction input endpoint is outside the NCCL topology");
    }
  }

  struct endpoint_buffer_t {
    distributed_endpoint_t endpoint;
    value_type *dense = nullptr;
    uint8_t *flags = nullptr;
  };

  index_t OutputElements() const {
    index_t total = 1;
    for (int dim = 0; dim < OutputRank; ++dim)
      total *= output_shape_[dim];
    return total;
  }

  index_t ReducedElements() const {
    index_t total = 1;
    for (int i = 0; i < D; ++i)
      total *= input_.Size(axes_[i]);
    return total;
  }

  std::vector<index_t> BuildMapping(
      size_t distribution_index,
      const cuda::std::array<index_t, OutputRank> &partial_shape) const {
    index_t count = 1;
    for (int dim = 0; dim < OutputRank; ++dim)
      count *= partial_shape[dim];
    std::vector<index_t> mapping(static_cast<size_t>(count));
    std::array<bool, InputRank> reduced{};
    for (int axis : axes_)
      reduced[axis] = true;
    for (index_t linear = 0; linear < count; ++linear) {
      const auto partial_index =
          DistributedUnflatten<OutputRank>(linear, partial_shape);
      distributed_index_t<InputRank> local_index{};
      int partial_dim = 0;
      for (int dim = 0; dim < InputRank; ++dim)
        if (!reduced[dim])
          local_index[dim] = partial_index[partial_dim++];
      const auto global = input_.DistributionDescriptor().LocalToGlobal(
          distribution_index, local_index);
      cuda::std::array<index_t, OutputRank> output_index{};
      int output_dim = 0;
      for (int dim = 0; dim < InputRank; ++dim)
        if (!reduced[dim])
          output_index[output_dim++] = global[dim];
      mapping[static_cast<size_t>(linear)] =
          DistributedFlatten<OutputRank>(output_index, output_shape_);
    }
    return mapping;
  }

  template <typename Out, typename Executor>
  void ExecuteNccl(Out &out, Executor &executor) const {
#ifndef __CUDACC__
    MATX_THROW(matxNotSupported,
               "Distributed reductions must be compiled by a CUDA compiler");
#else
    constexpr int threads = 256;
    const index_t output_elements = OutputElements();
    const auto output_blocks =
        static_cast<unsigned int>((output_elements + threads - 1) / threads);
    std::vector<endpoint_buffer_t> buffers;
    buffers.reserve(out.LocalFragmentCount());
    for (size_t local = 0; local < out.LocalFragmentCount(); ++local) {
      const auto endpoint = out.LocalFragment(local).endpoint;
      executor.ForEndpoint(endpoint, [&](cudaExecutor &local_executor) {
        endpoint_buffer_t buffer{endpoint};
        MATX_CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void **>(&buffer.dense),
            output_elements * sizeof(value_type), local_executor.getStream()));
        DistributedInitializeKernel<KIND>
            <<<output_blocks, threads, 0, local_executor.getStream()>>>(
                buffer.dense, output_elements);
        MATX_CUDA_CHECK(cudaGetLastError());
        buffers.push_back(buffer);
      });
    }

    for (size_t local = 0; local < input_.LocalFragmentCount(); ++local) {
      const auto &fragment = input_.LocalFragment(local);
      if (fragment.view.TotalSize() == 0) {
        continue;
      }
      auto buffer_it =
          std::find_if(buffers.begin(), buffers.end(), [&](const auto &buffer) {
            return buffer.endpoint == fragment.endpoint;
          });
      matx::detail::DistributedCheck(buffer_it != buffers.end(),
                                     matxInvalidParameter,
                                     "Input endpoint has no reduction replica");
      executor.ForEndpoint(
          fragment.endpoint, [&](cudaExecutor &local_executor) {
            cuda::std::array<index_t, OutputRank> partial_shape{};
            std::array<bool, InputRank> reduced{};
            for (int axis : axes_)
              reduced[axis] = true;
            int partial_dim = 0;
            const auto local_shape = input_.DistributionDescriptor().LocalShape(
                fragment.distribution_index);
            for (int dim = 0; dim < InputRank; ++dim)
              if (!reduced[dim])
                partial_shape[partial_dim++] = local_shape[dim];
            index_t partial_elements = 1;
            for (int dim = 0; dim < OutputRank; ++dim)
              partial_elements *= partial_shape[dim];
            const auto partial_blocks = static_cast<unsigned int>(
                (partial_elements + threads - 1) / threads);
            value_type *partial_data = nullptr;
            index_t *device_mapping = nullptr;
            MATX_CUDA_CHECK(
                cudaMallocAsync(reinterpret_cast<void **>(&partial_data),
                                partial_elements * sizeof(value_type),
                                local_executor.getStream()));
            auto partial = make_tensor(partial_data, partial_shape, false);
            RunLocalReduction<KIND, D>(partial, fragment.view, local_executor,
                                       axes_);
            const auto host_mapping =
                BuildMapping(fragment.distribution_index, partial_shape);
            MATX_CUDA_CHECK(
                cudaMallocAsync(reinterpret_cast<void **>(&device_mapping),
                                partial_elements * sizeof(index_t),
                                local_executor.getStream()));
            MATX_CUDA_CHECK(cudaMemcpyAsync(device_mapping, host_mapping.data(),
                                            partial_elements * sizeof(index_t),
                                            cudaMemcpyHostToDevice,
                                            local_executor.getStream()));
            DistributedScatterKernel<KIND>
                <<<partial_blocks, threads, 0, local_executor.getStream()>>>(
                    buffer_it->dense, partial_data, device_mapping,
                    partial_elements);
            MATX_CUDA_CHECK(cudaGetLastError());
            MATX_CUDA_CHECK(
                cudaFreeAsync(device_mapping, local_executor.getStream()));
            MATX_CUDA_CHECK(
                cudaFreeAsync(partial_data, local_executor.getStream()));
          });
    }

    constexpr bool logical = KIND == distributed_reduction_kind::all ||
                             KIND == distributed_reduction_kind::any;
    if constexpr (logical) {
      for (auto &buffer : buffers)
        executor.ForEndpoint(
            buffer.endpoint, [&](cudaExecutor &local_executor) {
              MATX_CUDA_CHECK(
                  cudaMallocAsync(reinterpret_cast<void **>(&buffer.flags),
                                  output_elements, local_executor.getStream()));
              DistributedToFlagsKernel<<<output_blocks, threads, 0,
                                         local_executor.getStream()>>>(
                  buffer.flags, buffer.dense, output_elements);
              MATX_CUDA_CHECK(cudaGetLastError());
            });
    }

    constexpr bool replicated_input = is_replicated_distribution_v<
        typename remove_cvref_t<Input>::distribution_type>;
    if constexpr (!replicated_input) {
      matx::detail::NcclCheck(ncclGroupStart(), "ncclGroupStart");
      for (auto &buffer : buffers) {
        matx::detail::distributed_device_guard guard{buffer.endpoint.device_id};
        auto local_executor = executor.LocalExecutor(buffer.endpoint);
        if constexpr (logical) {
          matx::detail::NcclCheck(
              ncclAllReduce(
                  buffer.flags, buffer.flags, output_elements, ncclUint8,
                  KIND == distributed_reduction_kind::all ? ncclMin : ncclMax,
                  executor.NcclCommunicator(buffer.endpoint),
                  local_executor.getStream()),
              "ncclAllReduce");
        } else {
          ncclRedOp_t operation = ncclSum;
          if constexpr (KIND == distributed_reduction_kind::prod)
            operation = ncclProd;
          else if constexpr (KIND == distributed_reduction_kind::min)
            operation = ncclMin;
          else if constexpr (KIND == distributed_reduction_kind::max)
            operation = ncclMax;
          using storage_type = typename nccl_storage_type<value_type>::type;
          const size_t count = is_complex_v<value_type>
                                   ? static_cast<size_t>(output_elements) * 2
                                   : static_cast<size_t>(output_elements);
          matx::detail::NcclCheck(
              ncclAllReduce(buffer.dense, buffer.dense, count,
                            DistributedNcclType<storage_type>(), operation,
                            executor.NcclCommunicator(buffer.endpoint),
                            local_executor.getStream()),
              "ncclAllReduce");
        }
      }
      matx::detail::NcclCheck(ncclGroupEnd(), "ncclGroupEnd");
    }

    for (auto &buffer : buffers)
      executor.ForEndpoint(buffer.endpoint, [&](cudaExecutor &local_executor) {
        if constexpr (logical) {
          DistributedFromFlagsKernel<<<output_blocks, threads, 0,
                                       local_executor.getStream()>>>(
              buffer.dense, buffer.flags, output_elements);
          MATX_CUDA_CHECK(cudaGetLastError());
        }
        if constexpr (KIND == distributed_reduction_kind::mean) {
          DistributedMeanScaleKernel<<<output_blocks, threads, 0,
                                       local_executor.getStream()>>>(
              buffer.dense, output_elements, ReducedElements());
          MATX_CUDA_CHECK(cudaGetLastError());
        }
        auto dense = make_tensor(buffer.dense, output_shape_, false);
        const auto &endpoints = executor.CollectiveEndpoints();
        const size_t distribution_index = static_cast<size_t>(
            std::find(endpoints.begin(), endpoints.end(), buffer.endpoint) -
            endpoints.begin());
        auto &local_out =
            out.LocalFragmentForDistributionIndex(distribution_index).view;
        (local_out = dense).run(local_executor);
        if (buffer.flags != nullptr)
          MATX_CUDA_CHECK(
              cudaFreeAsync(buffer.flags, local_executor.getStream()));
        MATX_CUDA_CHECK(
            cudaFreeAsync(buffer.dense, local_executor.getStream()));
      });
#endif
  }
#endif

  Input input_;
  cuda::std::array<int, D> axes_{};
  cuda::std::array<index_t, OutputRank> output_shape_{};
};

template <distributed_reduction_kind KIND, typename T, int RANK,
          typename Distribution, int D>
auto MakeDistributedReduction(
    const distributed_tensor_t<T, RANK, Distribution> &input,
    const int (&dims)[D]) {
  return distributed_reduction_op<
      KIND, distributed_tensor_t<T, RANK, Distribution>, D>{input, dims};
}

template <distributed_reduction_kind KIND, typename T, int RANK,
          typename Distribution>
auto MakeDistributedReduction(
    const distributed_tensor_t<T, RANK, Distribution> &input) {
  return distributed_reduction_op<
      KIND, distributed_tensor_t<T, RANK, Distribution>, RANK>{input};
}

} // namespace matx::experimental::detail

namespace matx {

#define MATX_DISTRIBUTED_REDUCTION_OVERLOAD(NAME, KIND)                        \
  template <typename T, int RANK, typename Distribution, int D>                \
  auto NAME(                                                                   \
      const experimental::distributed_tensor_t<T, RANK, Distribution> &input,  \
      const int(&dims)[D]) {                                                   \
    return experimental::detail::MakeDistributedReduction<                     \
        experimental::detail::distributed_reduction_kind::KIND>(input, dims);  \
  }                                                                            \
  template <typename T, int RANK, typename Distribution>                       \
  auto NAME(const experimental::distributed_tensor_t<T, RANK, Distribution>    \
                &input) {                                                      \
    return experimental::detail::MakeDistributedReduction<                     \
        experimental::detail::distributed_reduction_kind::KIND>(input);        \
  }

MATX_DISTRIBUTED_REDUCTION_OVERLOAD(sum, sum)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(prod, prod)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(min, min)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(max, max)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(mean, mean)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(all, all)
MATX_DISTRIBUTED_REDUCTION_OVERLOAD(any, any)

#undef MATX_DISTRIBUTED_REDUCTION_OVERLOAD

} // namespace matx
