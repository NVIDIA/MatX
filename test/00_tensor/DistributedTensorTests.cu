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

#include "matx.h"
#include "gtest/gtest.h"

#include <cmath>
#include <type_traits>
#include <vector>

using namespace matx;
using namespace matx::experimental;

namespace {

struct AddValues {
  __host__ __device__ float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

struct AddIntegerValues {
  __host__ __device__ float operator()(float lhs, int rhs) const {
    return lhs + static_cast<float>(rhs);
  }
};

struct BlackScholesValues {
  __host__ __device__ float operator()(float strike, float spot,
                                       float volatility, float rate,
                                       float time) const {
    const float volatility_sqrt_time = volatility * sqrtf(time);
    const float d1 =
        (logf(spot / strike) + (rate + 0.5F * volatility * volatility) * time) /
        volatility_sqrt_time;
    const float d2 = d1 - volatility_sqrt_time;
    constexpr float inv_sqrt_two = 0.7071067811865475244F;
    const float cdf_d1 = 0.5F * (1.0F + erff(d1 * inv_sqrt_two));
    const float cdf_d2 = 0.5F * (1.0F + erff(d2 * inv_sqrt_two));
    return spot * cdf_d1 - strike * expf(-rate * time) * cdf_d2;
  }
};

template <typename Distributed, typename Regular>
concept CanAssignRegularToDistributed = requires(
    Distributed &distributed, Regular &regular) { distributed = regular; };

static_assert(!CanAssignRegularToDistributed<
              distributed_tensor_t<float, 1, block_distribution_t<1>>,
              tensor_t<float, 1>>);
static_assert(!cuda::std::is_constructible_v<
              local_fragment_t<float, 1>, tensor_t<double, 1>, size_t,
              distributed_endpoint_t, std::shared_ptr<void>>);
static_assert(!cuda::std::is_constructible_v<
              local_fragment_t<float, 1>, tensor_t<float, 2>, size_t,
              distributed_endpoint_t, std::shared_ptr<void>>);

std::vector<distributed_endpoint_t>
MakeEndpoints(const std::vector<int> &devices) {
  std::vector<distributed_endpoint_t> endpoints;
  endpoints.reserve(devices.size());
  for (int device : devices) {
    endpoints.push_back({0, device});
  }
  return endpoints;
}

template <typename Distributed, typename Generator>
void FillDistributed(Distributed &tensor, Generator &&generator) {
  const auto &distribution = tensor.DistributionDescriptor();
  for (size_t local = 0; local < tensor.LocalFragmentCount(); ++local) {
    const auto &fragment = tensor.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    const index_t count = tensor.LocalView(local).TotalSize();
    std::vector<typename Distributed::value_type> host(
        static_cast<size_t>(count));
    for (index_t linear = 0; linear < count; ++linear) {
      const auto local_index =
          experimental::detail::DistributedUnflatten<Distributed::Rank()>(
              linear, shape);
      const auto global_index =
          distribution.LocalToGlobal(fragment.distribution_index, local_index);
      host[static_cast<size_t>(linear)] = generator(global_index);
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(tensor.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(host[0]),
                               cudaMemcpyDefault));
  }
}

void RunBlackScholes(const std::vector<int> &devices) {
  constexpr index_t count = 257;
  distributed_context context{devices};
  distributedCUDAExecutor executor{context};
  auto distribution =
      block_distribution_t<1>::Slab({count}, MakeEndpoints(devices));

  auto strike = make_distributed_tensor<float>(distribution, context);
  auto spot = make_distributed_tensor<float>(distribution, context);
  auto volatility = make_distributed_tensor<float>(distribution, context);
  auto rate = make_distributed_tensor<float>(distribution, context);
  auto time = make_distributed_tensor<float>(distribution, context);
  auto distributed_output =
      make_distributed_tensor<float>(distribution, context);

  FillDistributed(strike, [](const auto &index) {
    return 80.0F + static_cast<float>(index[0] % 17);
  });
  FillDistributed(spot, [](const auto &index) {
    return 90.0F + static_cast<float>(index[0] % 23);
  });
  FillDistributed(volatility, [](const auto &index) {
    return 0.15F + 0.001F * static_cast<float>(index[0] % 11);
  });
  FillDistributed(rate, [](const auto &index) {
    return 0.01F + 0.0001F * static_cast<float>(index[0] % 7);
  });
  FillDistributed(time, [](const auto &index) {
    return 0.5F + 0.01F * static_cast<float>(index[0] % 19);
  });

  auto regular_strike = make_tensor<float>({count});
  auto regular_spot = make_tensor<float>({count});
  auto regular_volatility = make_tensor<float>({count});
  auto regular_rate = make_tensor<float>({count});
  auto regular_time = make_tensor<float>({count});
  auto reference = make_tensor<float>({count});
  for (index_t i = 0; i < count; ++i) {
    regular_strike(i) = 80.0F + static_cast<float>(i % 17);
    regular_spot(i) = 90.0F + static_cast<float>(i % 23);
    regular_volatility(i) = 0.15F + 0.001F * static_cast<float>(i % 11);
    regular_rate(i) = 0.01F + 0.0001F * static_cast<float>(i % 7);
    regular_time(i) = 0.5F + 0.01F * static_cast<float>(i % 19);
  }
  cudaExecutor reference_executor{};
  (reference = matx::apply(BlackScholesValues{}, regular_strike, regular_spot,
                           regular_volatility, regular_rate, regular_time))
      .run(reference_executor);
  reference_executor.sync();

  (distributed_output =
       apply(BlackScholesValues{}, strike, spot, volatility, rate, time))
      .run(executor);
  executor.sync();
  EXPECT_EQ(executor.TransferCount(), 0U);

  auto regular_output = make_tensor<float>({count});
  (regular_output = distributed_output).run(executor);
  executor.sync();
  EXPECT_GT(executor.TransferCount(), 0U);

  for (index_t i = 0; i < count; ++i) {
    EXPECT_NEAR(regular_output(i), reference(i), 2.0e-5F)
        << "global index " << i;
  }
}

} // namespace

TEST(DistributedTensor, UnevenSlabMapping) {
  const auto distribution =
      block_distribution_t<1>::Slab({10}, {{0, 0}, {0, 1}, {0, 2}});
  EXPECT_EQ(distribution.LocalShape(0)[0], 4);
  EXPECT_EQ(distribution.LocalShape(1)[0], 3);
  EXPECT_EQ(distribution.LocalShape(2)[0], 3);
  EXPECT_EQ(distribution.LocalToGlobal(0, {3})[0], 3);
  EXPECT_EQ(distribution.LocalToGlobal(1, {0})[0], 4);
  EXPECT_EQ(distribution.LocalToGlobal(2, {2})[0], 9);
}

TEST(DistributedTensor, ReplicatedMapping) {
  replicated_distribution_t<2> distribution{{3, 5}, {{0, 0}, {0, 1}}};
  EXPECT_EQ(distribution.FragmentCount(), 2U);
  EXPECT_EQ(distribution.LocalShape(0), (distributed_index_t<2>{3, 5}));
  EXPECT_EQ(distribution.LocalShape(1), (distributed_index_t<2>{3, 5}));
  EXPECT_EQ(distribution.LocalToGlobal(1, {2, 4}),
            (distributed_index_t<2>{2, 4}));
  EXPECT_THROW((replicated_distribution_t<1>{{8}, {{0, 0}, {0, 0}}}),
               matx::detail::matxException);
}

TEST(DistributedTensor, RejectsOverlapAndGap) {
  using descriptor = block_fragment_descriptor_t<1>;
  EXPECT_THROW(
      (block_distribution_t<1>{
          {8},
          std::vector<descriptor>{{{0, 0}, {0}, {5}}, {{0, 1}, {4}, {4}}}}),
      matx::detail::matxException);
  EXPECT_THROW(
      (block_distribution_t<1>{
          {8},
          std::vector<descriptor>{{{0, 0}, {0}, {3}}, {{0, 1}, {4}, {4}}}}),
      matx::detail::matxException);
}

TEST(DistributedTensor, BlockCyclicMapping) {
  block_cyclic_distribution_t distribution{
      {7, 6}, {2, 2}, {2, 2}, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
  EXPECT_EQ(distribution.LocalShape(0)[0], 4);
  EXPECT_EQ(distribution.LocalShape(0)[1], 4);
  EXPECT_EQ(distribution.LocalShape(3)[0], 3);
  EXPECT_EQ(distribution.LocalShape(3)[1], 2);
  EXPECT_EQ(distribution.LocalToGlobal(3, {0, 0})[0], 2);
  EXPECT_EQ(distribution.LocalToGlobal(3, {0, 0})[1], 2);
  EXPECT_EQ(distribution.LocalToGlobal(0, {2, 2})[0], 4);
  EXPECT_EQ(distribution.LocalToGlobal(0, {2, 2})[1], 4);

  block_cyclic_distribution_t column_major{
      {8, 12},
      {2, 2},
      {2, 3},
      {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}},
      distributed_grid_layout::column_major};
  // Communicator rank 1 maps to process coordinate (1, 0), not (0, 1).
  EXPECT_EQ(column_major.LocalToGlobal(1, {0, 0})[0], 2);
  EXPECT_EQ(column_major.LocalToGlobal(1, {0, 0})[1], 0);
}

TEST(DistributedTensor, RegularOperatorsSelectBlockCyclicBackend) {
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  block_cyclic_distribution_t distribution{
      {4, 4}, {2, 2}, {1, 1}, {{0, 0}}};
  auto a = make_distributed_tensor<float>(distribution, context);
  auto b = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  auto matmul_op = matmul(a, b);
  auto chol_op = chol(a);
  static_assert(is_distributed_expression_v<decltype(matmul_op)>);
  static_assert(is_distributed_expression_v<decltype(chol_op)>);

  EXPECT_THROW((output = matmul_op).run(executor),
               matx::detail::matxException);
  EXPECT_THROW((output = chol_op).run(executor),
               matx::detail::matxException);
}

TEST(DistributedTensor, PointwiseCopyAndMaterialize) {
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({31}, {{0, 0}});
  auto lhs = make_distributed_tensor<float>(distribution, context);
  auto rhs = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  FillDistributed(
      lhs, [](const auto &index) { return static_cast<float>(index[0]); });
  FillDistributed(
      rhs, [](const auto &index) { return static_cast<float>(2 * index[0]); });
  (output = apply(AddValues{}, lhs, rhs)).run(executor);

  auto copied = make_distributed_tensor<float>(distribution, context);
  (copied = output).run(executor);

  auto regular = make_tensor<float>({31});
  (regular = copied).run(executor);
  executor.sync();
  for (index_t i = 0; i < regular.Size(0); ++i) {
    EXPECT_FLOAT_EQ(regular(i), static_cast<float>(3 * i));
  }
}

TEST(DistributedTensor, WrapsExistingLocalPointers) {
  constexpr index_t count = 31;
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({count}, {{0, 0}});
  auto input_storage = make_tensor<float>({count}, MATX_DEVICE_MEMORY);
  auto output_storage = make_tensor<float>({count}, MATX_DEVICE_MEMORY);

  auto input =
      make_distributed_tensor(distribution, context, input_storage.Data());
  auto output = make_distributed_tensor(
      distribution, context, std::vector<float *>{output_storage.Data()});
  EXPECT_EQ(input.LocalView(0).Data(), input_storage.Data());
  EXPECT_EQ(output.LocalView(0).Data(), output_storage.Data());

  FillDistributed(
      input, [](const auto &index) { return static_cast<float>(index[0]); });
  (output = apply(AddValues{}, input, input)).run(executor);
  executor.sync();

  std::vector<float> host(static_cast<size_t>(count));
  MATX_CUDA_CHECK(cudaMemcpy(host.data(), output_storage.Data(),
                             host.size() * sizeof(float),
                             cudaMemcpyDefault));
  for (index_t i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(host[static_cast<size_t>(i)], 2.0F * static_cast<float>(i));
  }

  EXPECT_THROW(
      (void)make_distributed_tensor(
          distribution, context,
          std::vector<distributed_local_pointer_t<float>>{{0, nullptr, {}}}),
      matx::detail::matxException);
  EXPECT_THROW(
      (void)make_distributed_tensor(
          distribution, context,
          std::vector<float *>{input_storage.Data(), output_storage.Data()}),
      matx::detail::matxException);
}

TEST(DistributedTensor, PointwiseInputsMayHaveDifferentElementTypes) {
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({13}, {{0, 0}});
  auto floats = make_distributed_tensor<float>(distribution, context);
  auto integers = make_distributed_tensor<int>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  FillDistributed(floats, [](const auto &index) {
    return 0.5F * static_cast<float>(index[0]);
  });
  FillDistributed(integers,
                  [](const auto &index) { return static_cast<int>(index[0]); });
  (output = apply(AddIntegerValues{}, floats, integers))
      .run(executor);

  auto regular = make_tensor<float>({13});
  (regular = output).run(executor);
  executor.sync();
  for (index_t i = 0; i < regular.Size(0); ++i) {
    EXPECT_FLOAT_EQ(regular(i), 1.5F * static_cast<float>(i));
  }
}

TEST(DistributedTensor, MaterializesToStridedRegularView) {
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({17}, {{0, 0}});
  auto source = make_distributed_tensor<float>(distribution, context);
  FillDistributed(source, [](const auto &index) {
    return 10.0F + static_cast<float>(index[0]);
  });

  auto host_storage = make_tensor<float>({34}, MATX_HOST_MEMORY);
  for (index_t i = 0; i < host_storage.Size(0); ++i) {
    host_storage(i) = -1.0F;
  }
  auto host_strided = slice(host_storage, {0}, {matxEnd}, {2});
  (host_strided = source).run(executor);
  executor.sync();

  for (index_t i = 0; i < 17; ++i) {
    EXPECT_FLOAT_EQ(host_storage(2 * i), 10.0F + static_cast<float>(i));
    EXPECT_FLOAT_EQ(host_storage(2 * i + 1), -1.0F);
  }

  auto device_storage = make_tensor<float>({34}, MATX_DEVICE_MEMORY);
  cudaExecutor local_executor{};
  (device_storage = -1.0F).run(local_executor);
  local_executor.sync();
  auto device_strided = slice(device_storage, {0}, {matxEnd}, {2});
  (device_strided = source).run(executor);
  executor.sync();

  std::vector<float> device_result(34);
  MATX_CUDA_CHECK(cudaMemcpy(device_result.data(), device_storage.Data(),
                             device_result.size() * sizeof(float),
                             cudaMemcpyDefault));
  for (index_t i = 0; i < 17; ++i) {
    EXPECT_FLOAT_EQ(device_result[static_cast<size_t>(2 * i)],
                    10.0F + static_cast<float>(i));
    EXPECT_FLOAT_EQ(device_result[static_cast<size_t>(2 * i + 1)], -1.0F);
  }
}

TEST(DistributedTensor, MaterializesToContiguousHostAndDevice) {
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({19}, {{0, 0}});
  auto source = make_distributed_tensor<float>(distribution, context);
  FillDistributed(source, [](const auto &index) {
    return 20.0F + static_cast<float>(index[0]);
  });

  auto host = make_tensor<float>({19}, MATX_HOST_MEMORY);
  (host = source).run(executor);
  executor.sync();
  for (index_t i = 0; i < 19; ++i) {
    EXPECT_FLOAT_EQ(host(i), 20.0F + static_cast<float>(i));
  }

  auto device = make_tensor<float>({19}, MATX_DEVICE_MEMORY);
  (device = source).run(executor);
  executor.sync();
  std::vector<float> device_result(19);
  MATX_CUDA_CHECK(cudaMemcpy(device_result.data(), device.Data(),
                             device_result.size() * sizeof(float),
                             cudaMemcpyDefault));
  for (index_t i = 0; i < 19; ++i) {
    EXPECT_FLOAT_EQ(device_result[static_cast<size_t>(i)],
                    20.0F + static_cast<float>(i));
  }
}

TEST(DistributedTensor, RejectsContextMismatchAndAliasing) {
  distributed_context context_a{{0}};
  distributed_context context_b{{0}};
  distributedCUDAExecutor executor{context_a};
  auto distribution = block_distribution_t<1>::Slab({8}, {{0, 0}});
  auto a = make_distributed_tensor<float>(distribution, context_a);
  auto b = make_distributed_tensor<float>(distribution, context_b);
  EXPECT_THROW((void)apply(AddValues{}, a, b),
               matx::detail::matxException);

  using descriptor = block_fragment_descriptor_t<1>;
  block_distribution_t<1> different_layout{
      {8}, std::vector<descriptor>{{{0, 0}, {0}, {3}}, {{0, 0}, {3}, {5}}}};
  auto differently_partitioned =
      make_distributed_tensor<float>(different_layout, context_a);
  EXPECT_THROW((void)apply(AddValues{}, a, differently_partitioned),
               matx::detail::matxException);

  auto regular = make_tensor<float>({8});
  std::vector<local_fragment_t<float, 1>> fragments;
  fragments.push_back({regular, 0, {0, 0}, {}});
  distributed_tensor_t<float, 1, block_distribution_t<1>> aliasing_source{
      distribution, context_a, std::move(fragments)};
  EXPECT_THROW((regular = aliasing_source).run(executor),
               matx::detail::matxException);
}

TEST(DistributedTensor, BlackScholesOneGpu) { RunBlackScholes({0}); }
