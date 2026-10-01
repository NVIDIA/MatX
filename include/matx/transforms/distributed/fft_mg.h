////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <cufft.h>
#include <cufftXt.h>

#include "matx/core/distributed_tensor.h"
#include "matx/core/operator_options.h"

namespace matx::experimental {
namespace detail {

inline void CufftMgCheck(cufftResult status, const char *operation) {
  if (status != CUFFT_SUCCESS) {
    MATX_THROW(matxCufftError, std::string(operation) +
                                   " failed with cuFFT status " +
                                   std::to_string(static_cast<int>(status)));
  }
}

class cufft_mg_plan {
public:
  cufft_mg_plan() { CufftMgCheck(cufftCreate(&plan_), "cufftCreate"); }
  cufft_mg_plan(const cufft_mg_plan &) = delete;
  ~cufft_mg_plan() {
    if (input_ != nullptr) {
      (void)cufftXtFree(input_);
    }
    if (output_ != nullptr) {
      (void)cufftXtFree(output_);
    }
    if (plan_ != 0) {
      (void)cufftDestroy(plan_);
    }
  }

  cufftHandle Handle() const noexcept { return plan_; }
  cudaLibXtDesc **InputAddress() noexcept { return &input_; }
  cudaLibXtDesc **OutputAddress() noexcept { return &output_; }
  cudaLibXtDesc *Input() const noexcept { return input_; }
  cudaLibXtDesc *Output() const noexcept { return output_; }

private:
  cufftHandle plan_ = 0;
  cudaLibXtDesc *input_ = nullptr;
  cudaLibXtDesc *output_ = nullptr;
};

template <typename T> class pinned_buffer {
public:
  explicit pinned_buffer(size_t count) : count_{count} {
    MATX_CUDA_CHECK(
        cudaMallocHost(reinterpret_cast<void **>(&data_), count * sizeof(T)));
  }
  pinned_buffer(const pinned_buffer &) = delete;
  ~pinned_buffer() {
    if (data_ != nullptr) {
      (void)cudaFreeHost(data_);
    }
  }
  T *Data() noexcept { return data_; }
  const T *Data() const noexcept { return data_; }
  size_t Size() const noexcept { return count_; }

private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

template <typename Tensor>
void GatherMgInput(const Tensor &tensor,
                   pinned_buffer<typename Tensor::value_type> &host) {
  const auto &distribution = tensor.DistributionDescriptor();
  for (size_t fragment = 0; fragment < distribution.FragmentCount();
       ++fragment) {
    const auto local_zero = distributed_index_t<1>{0};
    const index_t origin = distribution.LocalToGlobal(fragment, local_zero)[0];
    const auto &local = tensor.LocalFragmentForDistributionIndex(fragment).view;
    matx::detail::distributed_device_guard guard{
        distribution.FragmentEndpoint(fragment).device_id};
    MATX_CUDA_CHECK(cudaMemcpy(host.Data() + origin, local.Data(),
                               static_cast<size_t>(local.Size(0)) *
                                   sizeof(typename Tensor::value_type),
                               cudaMemcpyDeviceToHost));
  }
}

template <typename Tensor>
void ScatterMgOutput(Tensor &tensor,
                     const pinned_buffer<typename Tensor::value_type> &host) {
  const auto &distribution = tensor.DistributionDescriptor();
  for (size_t fragment = 0; fragment < distribution.FragmentCount();
       ++fragment) {
    const auto local_zero = distributed_index_t<1>{0};
    const index_t origin = distribution.LocalToGlobal(fragment, local_zero)[0];
    auto &local = tensor.LocalFragmentForDistributionIndex(fragment).view;
    matx::detail::distributed_device_guard guard{
        distribution.FragmentEndpoint(fragment).device_id};
    MATX_CUDA_CHECK(cudaMemcpy(local.Data(), host.Data() + origin,
                               static_cast<size_t>(local.Size(0)) *
                                   sizeof(typename Tensor::value_type),
                               cudaMemcpyHostToDevice));
  }
}

template <matx::detail::FFTDirection Direction, typename Out, typename In,
          typename Executor>
void FftMgImpl(Out &out, const In &in, Executor &executor, FFTNorm norm) {
  using value_type = typename remove_cvref_t<Out>::value_type;
  static_assert(
      is_distributed_tensor_v<Out> && is_distributed_tensor_v<In> &&
          remove_cvref_t<Out>::Rank() == 1 && remove_cvref_t<In>::Rank() == 1,
      "cuFFT multi-GPU currently supports rank-1 distributed tensors");
  static_assert(
      std::is_same_v<typename remove_cvref_t<Out>::distribution_type,
                     block_distribution_t<1>> &&
          std::is_same_v<typename remove_cvref_t<In>::distribution_type,
                         block_distribution_t<1>>,
      "cuFFT multi-GPU requires block-distributed tensors");
  static_assert(
      std::is_same_v<value_type, typename remove_cvref_t<In>::value_type> &&
          (std::is_same_v<value_type, cuda::std::complex<float>> ||
           std::is_same_v<value_type, cuda::std::complex<double>>),
      "cuFFT multi-GPU supports complex<float> and complex<double>");
  static_assert(sizeof(cuda::std::complex<float>) == sizeof(cufftComplex));
  static_assert(sizeof(cuda::std::complex<double>) ==
                sizeof(cufftDoubleComplex));

  matx::detail::DistributedCheck(
      executor.Context().ProcessCount() == 1, matxNotSupported,
      "cuFFT multi-GPU (Mg/Xt) is a single-process backend");
  matx::detail::DistributedCheck(
      in.ContextId() == executor.ContextId() &&
          out.ContextId() == executor.ContextId(),
      matxInvalidExecutor,
      "cuFFT multi-GPU operands and executor contexts do not match");
  matx::detail::DistributedCheck(
      in.Size(0) == out.Size(0) &&
          in.DistributionDescriptor().Compatible(out.DistributionDescriptor()),
      matxInvalidSize,
      "cuFFT multi-GPU input and output distributions must match");
  matx::detail::DistributedCheck(
      in.LocalFragmentCount() >= 2, matxInvalidSize,
      "cuFFT multi-GPU requires at least two local GPUs");
  matx::detail::DistributedCheck(
      in.Size(0) <= static_cast<index_t>(std::numeric_limits<int>::max()),
      matxInvalidSize, "cuFFT multi-GPU transform size exceeds its API limit");
  matx::detail::DistributedCheck(
      in.LocalFragmentCount() == in.DistributionDescriptor().FragmentCount() &&
          out.LocalFragmentCount() ==
              out.DistributionDescriptor().FragmentCount(),
      matxInvalidParameter,
      "cuFFT multi-GPU requires every fragment to be local to this process");

  std::vector<int> devices;
  devices.reserve(in.LocalFragmentCount());
  for (size_t fragment = 0;
       fragment < in.DistributionDescriptor().FragmentCount(); ++fragment) {
    const int device =
        in.DistributionDescriptor().FragmentEndpoint(fragment).device_id;
    matx::detail::DistributedCheck(
        std::find(devices.begin(), devices.end(), device) == devices.end(),
        matxInvalidParameter,
        "cuFFT multi-GPU requires one fragment per distinct CUDA device");
    devices.push_back(device);
    matx::detail::DistributedCheck(
        in.LocalFragmentForDistributionIndex(fragment).view.IsContiguous() &&
            out.LocalFragmentForDistributionIndex(fragment).view.IsContiguous(),
        matxNotSupported,
        "cuFFT multi-GPU requires contiguous local fragments");
  }

  executor.sync();
  cufft_mg_plan plan;
  CufftMgCheck(cufftXtSetGPUs(plan.Handle(), static_cast<int>(devices.size()),
                              devices.data()),
               "cufftXtSetGPUs");
  std::vector<size_t> workspace_sizes(devices.size());
  const cufftType transform_type =
      std::is_same_v<value_type, cuda::std::complex<float>> ? CUFFT_C2C
                                                            : CUFFT_Z2Z;
  CufftMgCheck(cufftMakePlan1d(plan.Handle(), static_cast<int>(in.Size(0)),
                               transform_type, 1, workspace_sizes.data()),
               "cufftMakePlan1d");
  CufftMgCheck(cufftXtMalloc(plan.Handle(), plan.InputAddress(),
                             CUFFT_XT_FORMAT_INPLACE),
               "cufftXtMalloc(input)");
  CufftMgCheck(cufftXtMalloc(plan.Handle(), plan.OutputAddress(),
                             CUFFT_XT_FORMAT_INPLACE),
               "cufftXtMalloc(output)");

  pinned_buffer<value_type> host_input(static_cast<size_t>(in.Size(0)));
  pinned_buffer<value_type> host_output(static_cast<size_t>(out.Size(0)));
  GatherMgInput(in, host_input);
  CufftMgCheck(cufftXtMemcpy(plan.Handle(), plan.Input(), host_input.Data(),
                             CUFFT_COPY_HOST_TO_DEVICE),
               "cufftXtMemcpy(host-to-device)");
  if constexpr (std::is_same_v<value_type, cuda::std::complex<float>>) {
    CufftMgCheck(
        cufftXtExecDescriptorC2C(
            plan.Handle(), plan.Input(), plan.Output(),
            Direction == matx::detail::FFTDirection::FORWARD ? CUFFT_FORWARD
                                                             : CUFFT_INVERSE),
        "cufftXtExecDescriptorC2C");
  } else {
    CufftMgCheck(
        cufftXtExecDescriptorZ2Z(
            plan.Handle(), plan.Input(), plan.Output(),
            Direction == matx::detail::FFTDirection::FORWARD ? CUFFT_FORWARD
                                                             : CUFFT_INVERSE),
        "cufftXtExecDescriptorZ2Z");
  }
  CufftMgCheck(cufftXtMemcpy(plan.Handle(), host_output.Data(), plan.Output(),
                             CUFFT_COPY_DEVICE_TO_HOST),
               "cufftXtMemcpy(device-to-host)");

  double scale = 1.0;
  if (norm == FFTNorm::ORTHO) {
    scale = 1.0 / std::sqrt(static_cast<double>(in.Size(0)));
  } else if ((norm == FFTNorm::FORWARD &&
              Direction == matx::detail::FFTDirection::FORWARD) ||
             (norm == FFTNorm::BACKWARD &&
              Direction == matx::detail::FFTDirection::BACKWARD)) {
    scale = 1.0 / static_cast<double>(in.Size(0));
  }
  if (scale != 1.0) {
    for (size_t i = 0; i < host_output.Size(); ++i) {
      host_output.Data()[i] *=
          static_cast<typename value_type::value_type>(scale);
    }
  }
  ScatterMgOutput(out, host_output);
}

} // namespace detail
} // namespace matx::experimental
