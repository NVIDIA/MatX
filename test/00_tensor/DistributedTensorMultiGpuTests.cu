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
#include "matx/distributed.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace matx;
using namespace matx::experimental;

namespace {

struct MultiGpuBlackScholesValues {
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

struct IncrementValue {
  __host__ __device__ float operator()(float value) const {
    return value + 1.0F;
  }
};

template <typename Distributed, typename Generator>
void FillMultiGpu(Distributed &tensor, Generator &&generator) {
  static_assert(Distributed::Rank() == 1,
                "FillMultiGpu supports only rank-1 distributed tensors");
  const auto &distribution = tensor.DistributionDescriptor();
  for (size_t local = 0; local < tensor.LocalFragmentCount(); ++local) {
    const auto &fragment = tensor.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    std::vector<float> host(static_cast<size_t>(shape[0]));
    for (index_t i = 0; i < shape[0]; ++i) {
      const auto global =
          distribution.LocalToGlobal(fragment.distribution_index, {i});
      host[static_cast<size_t>(i)] = generator(global[0]);
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(tensor.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(float), cudaMemcpyDefault));
  }
}

} // namespace

TEST(DistributedTensorMultiGpu, CufftMgOneDimensionalComplex) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This integration test requires two CUDA devices";
  }

  constexpr index_t count = 1024;
  using complex_type = cuda::std::complex<float>;
  distributed_context context{{0, 1}};
  distributedCUDAExecutor executor{context};
  auto distribution =
      block_distribution_t<1>::Slab({count}, {{0, 0}, {0, 1}});
  auto input = make_distributed_tensor<complex_type>(distribution, context);
  auto output = make_distributed_tensor<complex_type>(distribution, context);

  for (size_t local = 0; local < input.LocalFragmentCount(); ++local) {
    const auto &fragment = input.LocalFragment(local);
    const index_t local_size = input.LocalView(local).Size(0);
    std::vector<complex_type> host(static_cast<size_t>(local_size),
                                   complex_type{0.0F, 0.0F});
    if (distribution.LocalToGlobal(fragment.distribution_index, {0})[0] == 0) {
      host[0] = complex_type{1.0F, 0.0F};
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(input.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(complex_type),
                               cudaMemcpyHostToDevice));
  }

  (output = fft(input)).run(executor);

  for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
    const auto &fragment = output.LocalFragment(local);
    const index_t local_size = output.LocalView(local).Size(0);
    std::vector<complex_type> host(static_cast<size_t>(local_size));
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(host.data(), output.LocalView(local).Data(),
                               host.size() * sizeof(complex_type),
                               cudaMemcpyDeviceToHost));
    for (index_t i = 0; i < local_size; ++i) {
      EXPECT_NEAR(host[static_cast<size_t>(i)].real(), 1.0F, 1.0e-5F);
      EXPECT_NEAR(host[static_cast<size_t>(i)].imag(), 0.0F, 1.0e-5F);
    }
  }
}

TEST(DistributedTensorMultiGpu, BlackScholesUnevenFragments) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This integration test requires two CUDA devices";
  }

  constexpr index_t count = 4099;
  distributed_context context{{0, 1}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<1>::Slab({count}, {{0, 0}, {0, 1}});

  auto strike = make_distributed_tensor<float>(distribution, context);
  auto spot = make_distributed_tensor<float>(distribution, context);
  auto volatility = make_distributed_tensor<float>(distribution, context);
  auto rate = make_distributed_tensor<float>(distribution, context);
  auto time = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  FillMultiGpu(strike, [](index_t i) { return 75.0F + float(i % 31); });
  FillMultiGpu(spot, [](index_t i) { return 85.0F + float(i % 37); });
  FillMultiGpu(volatility,
               [](index_t i) { return 0.12F + 0.001F * float(i % 13); });
  FillMultiGpu(rate, [](index_t i) { return 0.01F + 0.0001F * float(i % 11); });
  FillMultiGpu(time, [](index_t i) { return 0.25F + 0.01F * float(i % 29); });

  auto regular_strike = make_tensor<float>({count});
  auto regular_spot = make_tensor<float>({count});
  auto regular_volatility = make_tensor<float>({count});
  auto regular_rate = make_tensor<float>({count});
  auto regular_time = make_tensor<float>({count});
  auto reference = make_tensor<float>({count});
  for (index_t i = 0; i < count; ++i) {
    regular_strike(i) = 75.0F + float(i % 31);
    regular_spot(i) = 85.0F + float(i % 37);
    regular_volatility(i) = 0.12F + 0.001F * float(i % 13);
    regular_rate(i) = 0.01F + 0.0001F * float(i % 11);
    regular_time(i) = 0.25F + 0.01F * float(i % 29);
  }
  cudaExecutor reference_executor{};
  (reference =
       matx::apply(MultiGpuBlackScholesValues{}, regular_strike, regular_spot,
                   regular_volatility, regular_rate, regular_time))
      .run(reference_executor);
  reference_executor.sync();

  (output =
       apply(MultiGpuBlackScholesValues{}, strike, spot, volatility, rate, time))
      .run(executor);
  executor.sync();
  EXPECT_EQ(executor.TransferCount(), 0U);

  matx::detail::distributed_device_guard destination_guard{0};
  auto gathered = make_tensor<float>({count}, MATX_DEVICE_MEMORY);
  (gathered = output).run(executor);
  executor.sync();
  EXPECT_GT(executor.TransferCount(), 0U);

  std::vector<float> gathered_host(static_cast<size_t>(count));
  MATX_CUDA_CHECK(cudaMemcpy(gathered_host.data(), gathered.Data(),
                             gathered_host.size() * sizeof(float),
                             cudaMemcpyDefault));
  for (index_t i = 0; i < count; ++i) {
    EXPECT_NEAR(gathered_host[static_cast<size_t>(i)], reference(i), 2.0e-5F)
        << "global index " << i;
  }
}

TEST(DistributedTensorMultiGpu, BatchedMatmulEightGpuSharded) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 8) {
    GTEST_SKIP() << "This integration test requires eight CUDA devices";
  }

  constexpr int gpu_count = 8;
  constexpr index_t batch_count = 80;
  constexpr index_t matrix_size = 16;
  std::vector<int> devices;
  std::vector<distributed_endpoint_t> endpoints;
  for (int device = 0; device < gpu_count; ++device) {
    devices.push_back(device);
    endpoints.push_back({0, device});
  }

  distributed_context context{devices};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<3>::Slab(
      {batch_count, matrix_size, matrix_size}, endpoints, 0);
  auto a = make_distributed_tensor<float>(distribution, context);
  auto b = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  for (size_t local = 0; local < a.LocalFragmentCount(); ++local) {
    const auto &fragment = a.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    const size_t elements =
        static_cast<size_t>(shape[0] * shape[1] * shape[2]);
    std::vector<float> host_a(elements, 0.0F);
    std::vector<float> host_b(elements);
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0, 0});
      const float scale = static_cast<float>(global[0] + 1);
      for (index_t row = 0; row < matrix_size; ++row) {
        host_a[static_cast<size_t>(
            (batch * matrix_size + row) * matrix_size + row)] = scale;
        for (index_t column = 0; column < matrix_size; ++column) {
          host_b[static_cast<size_t>(
              (batch * matrix_size + row) * matrix_size + column)] =
              static_cast<float>(row * matrix_size + column + 1);
        }
      }
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(a.LocalView(local).Data(), host_a.data(),
                               elements * sizeof(float), cudaMemcpyDefault));
    MATX_CUDA_CHECK(cudaMemcpy(b.LocalView(local).Data(), host_b.data(),
                               elements * sizeof(float), cudaMemcpyDefault));
  }

  (output = matmul(a, b)).run(executor);
  executor.sync();
  EXPECT_EQ(executor.TransferCount(), 0U);

  for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
    const auto &fragment = output.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    const size_t elements =
        static_cast<size_t>(shape[0] * shape[1] * shape[2]);
    std::vector<float> host(elements);
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(host.data(), output.LocalView(local).Data(),
                               elements * sizeof(float), cudaMemcpyDefault));
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0, 0});
      const float scale = static_cast<float>(global[0] + 1);
      for (index_t row = 0; row < matrix_size; ++row) {
        for (index_t column = 0; column < matrix_size; ++column) {
          const float expected =
              scale * static_cast<float>(row * matrix_size + column + 1);
          EXPECT_NEAR(
              host[static_cast<size_t>(
                  (batch * matrix_size + row) * matrix_size + column)],
              expected, 1.0e-3F);
        }
      }
    }
  }
}

TEST(DistributedTensorMultiGpu, MatmulRejectsShardedMatrixDimensions) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This integration test requires two CUDA devices";
  }

  distributed_context context{{0, 1}};
  distributedCUDAExecutor executor{context};
  auto distribution =
      block_distribution_t<3>::Slab({4, 16, 16}, {{0, 0}, {0, 1}}, 1);
  auto a = make_distributed_tensor<float>(distribution, context);
  auto b = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  EXPECT_THROW((output = matmul(a, b)).run(executor),
               matx::detail::matxException);
}

TEST(DistributedTensorMultiGpu, BatchedCholTwoGpuSharded) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This integration test requires two CUDA devices";
  }

  constexpr index_t batch_count = 8;
  constexpr index_t matrix_size = 4;
  distributed_context context{{0, 1}};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<3>::Slab(
      {batch_count, matrix_size, matrix_size}, {{0, 0}, {0, 1}}, 0);
  auto input = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  for (size_t local = 0; local < input.LocalFragmentCount(); ++local) {
    const auto &fragment = input.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    const size_t elements =
        static_cast<size_t>(shape[0] * shape[1] * shape[2]);
    std::vector<float> host(elements, 0.0F);
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0, 0});
      for (index_t diagonal = 0; diagonal < matrix_size; ++diagonal) {
        const float root = static_cast<float>(global[0] + diagonal + 2);
        host[static_cast<size_t>(
            (batch * matrix_size + diagonal) * matrix_size + diagonal)] =
            root * root;
      }
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(input.LocalView(local).Data(), host.data(),
                               elements * sizeof(float), cudaMemcpyDefault));
  }

  (output = chol(input)).run(executor);
  executor.sync();
  EXPECT_EQ(executor.TransferCount(), 0U);

  for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
    const auto &fragment = output.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    const size_t elements =
        static_cast<size_t>(shape[0] * shape[1] * shape[2]);
    std::vector<float> host(elements);
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(host.data(), output.LocalView(local).Data(),
                               elements * sizeof(float), cudaMemcpyDefault));
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0, 0});
      for (index_t diagonal = 0; diagonal < matrix_size; ++diagonal) {
        EXPECT_NEAR(
            host[static_cast<size_t>(
                (batch * matrix_size + diagonal) * matrix_size + diagonal)],
            static_cast<float>(global[0] + diagonal + 2), 1.0e-4F);
      }
    }
  }
}

TEST(DistributedTensorMultiGpu, BatchedFftEightGpuSharded) {
  const char *gpu_count_env =
      std::getenv("MATX_DISTRIBUTED_FFT_GPU_COUNT");
  const int gpu_count =
      gpu_count_env == nullptr ? 8 : std::atoi(gpu_count_env);
  ASSERT_GT(gpu_count, 0);

  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < gpu_count) {
    GTEST_SKIP() << "This integration test requires " << gpu_count
                 << " CUDA devices";
  }

  using complex_type = cuda::std::complex<float>;
  constexpr index_t fft_size = 65536;
  constexpr unsigned long long input_bytes = 1ULL << 30;
  constexpr index_t total_elements = input_bytes / sizeof(complex_type);
  constexpr index_t total_batches = total_elements / fft_size;
  constexpr int warmup_iterations = 2;
  constexpr int timed_iterations = 5;
  static_assert(total_elements % fft_size == 0);
  ASSERT_GE(total_batches, gpu_count);

  std::vector<int> devices;
  std::vector<distributed_endpoint_t> endpoints;
  for (int device = 0; device < gpu_count; ++device) {
    devices.push_back(device);
    endpoints.push_back({0, device});
  }

  distributed_context context{devices};
  distributedCUDAExecutor executor{context};
  auto distribution = block_distribution_t<2>::Slab(
      {total_batches, fft_size}, endpoints, 0);
  auto input = make_distributed_tensor<complex_type>(distribution, context);
  auto output = make_distributed_tensor<complex_type>(distribution, context);
  ASSERT_EQ(input.LocalFragmentCount(), static_cast<size_t>(gpu_count));

  for (size_t local = 0; local < input.LocalFragmentCount(); ++local) {
    const auto &fragment = input.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    std::vector<complex_type> host(
        static_cast<size_t>(shape[0] * shape[1]));
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0});
      host[static_cast<size_t>(batch * fft_size)] =
          complex_type{static_cast<float>(global[0] + 1),
                       -static_cast<float>(global[0] + 1)};
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(input.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(complex_type),
                               cudaMemcpyDefault));
  }

  auto execute = [&]() { (output = fft(input)).run(executor); };
  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    execute();
  }
  executor.sync();

  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < timed_iterations; ++iteration) {
    execute();
  }
  executor.sync();
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  const double milliseconds = seconds / timed_iterations * 1.0e3;
  const double transforms_per_second =
      static_cast<double>(total_batches * timed_iterations) / seconds;
  std::cout << gpu_count << "-GPU batch-sharded FFT: " << total_batches
            << " batches x " << fft_size << " points, 1 GiB input, "
            << milliseconds << " ms/iteration, " << transforms_per_second
            << " transforms/s\n";

  for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
    const auto &fragment = output.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    std::vector<complex_type> first_bins(static_cast<size_t>(shape[0]));
    std::vector<complex_type> last_bins(static_cast<size_t>(shape[0]));
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    const size_t row_bytes =
        static_cast<size_t>(fft_size) * sizeof(complex_type);
    MATX_CUDA_CHECK(cudaMemcpy2D(
        first_bins.data(), sizeof(complex_type), output.LocalView(local).Data(),
        row_bytes, sizeof(complex_type), static_cast<size_t>(shape[0]),
        cudaMemcpyDefault));
    MATX_CUDA_CHECK(cudaMemcpy2D(
        last_bins.data(), sizeof(complex_type),
        output.LocalView(local).Data() + fft_size - 1, row_bytes,
        sizeof(complex_type), static_cast<size_t>(shape[0]),
        cudaMemcpyDefault));
    for (index_t batch = 0; batch < shape[0]; ++batch) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {batch, 0});
      const float expected = static_cast<float>(global[0] + 1);
      for (const auto value : {first_bins[static_cast<size_t>(batch)],
                               last_bins[static_cast<size_t>(batch)]}) {
        ASSERT_NEAR(value.real(), expected, 1.0e-5F)
            << "GPU " << fragment.endpoint.device_id << ", global batch "
            << global[0];
        ASSERT_NEAR(value.imag(), -expected, 1.0e-5F)
            << "GPU " << fragment.endpoint.device_id << ", global batch "
            << global[0];
      }
    }
  }
  EXPECT_EQ(executor.TransferCount(), 0U);
}

TEST(DistributedTensorMultiGpu, LargePointwiseEightGpu) {
  const char *gib_env = std::getenv("MATX_DISTRIBUTED_STRESS_GIB_PER_GPU");
  if (gib_env == nullptr) {
    GTEST_SKIP() << "Set MATX_DISTRIBUTED_STRESS_GIB_PER_GPU to run the "
                    "large eight-GPU stress test";
  }

  char *parse_end = nullptr;
  const unsigned long long gib_per_gpu = std::strtoull(gib_env, &parse_end, 10);
  ASSERT_NE(parse_end, gib_env);
  ASSERT_EQ(*parse_end, '\0');
  ASSERT_GT(gib_per_gpu, 0ULL);

  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 8) {
    GTEST_SKIP() << "This stress test requires eight visible CUDA devices";
  }

  constexpr int gpu_count = 8;
  constexpr int warmup_iterations = 3;
  constexpr int timed_iterations = 10;
  constexpr unsigned long long bytes_per_gib = 1ULL << 30;
  const auto elements_per_gpu =
      static_cast<index_t>(gib_per_gpu * bytes_per_gib / sizeof(float));
  ASSERT_GT(elements_per_gpu, 0);
  const index_t global_elements = elements_per_gpu * gpu_count;

  std::vector<int> devices;
  std::vector<distributed_endpoint_t> endpoints;
  for (int device = 0; device < gpu_count; ++device) {
    devices.push_back(device);
    endpoints.push_back({0, device});
  }

  distributed_context context{devices};
  distributedCUDAExecutor executor{context};
  auto distribution =
      block_distribution_t<1>::Slab({global_elements}, endpoints);
  auto values = make_distributed_tensor<float>(distribution, context);

  for (size_t local = 0; local < values.LocalFragmentCount(); ++local) {
    const auto &fragment = values.LocalFragment(local);
    executor.ForEndpoint(fragment.endpoint, [&](cudaExecutor &local_executor) {
      (values.LocalView(local) = 1.0F).run(local_executor);
    });
  }
  executor.sync();

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    (values = apply(IncrementValue{}, values)).run(executor);
  }
  executor.sync();

  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < timed_iterations; ++iteration) {
    (values = apply(IncrementValue{}, values)).run(executor);
  }
  executor.sync();
  const auto stop = std::chrono::steady_clock::now();

  const double seconds = std::chrono::duration<double>(stop - start).count();
  const double logical_gib = static_cast<double>(gib_per_gpu * gpu_count);
  const double traffic_gib =
      2.0 * logical_gib * static_cast<double>(timed_iterations);
  std::cout << "Eight-GPU distributed apply: " << logical_gib
            << " GiB logical tensor, " << seconds / timed_iterations * 1.0e3
            << " ms/iteration, " << traffic_gib / seconds
            << " GiB/s effective read+write bandwidth\n";

  const float expected =
      1.0F + static_cast<float>(warmup_iterations + timed_iterations);
  for (size_t local = 0; local < values.LocalFragmentCount(); ++local) {
    const auto &fragment = values.LocalFragment(local);
    const index_t local_size = values.LocalView(local).Size(0);
    float samples[2]{};
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(&samples[0], values.LocalView(local).Data(),
                               sizeof(float), cudaMemcpyDefault));
    MATX_CUDA_CHECK(cudaMemcpy(&samples[1],
                               values.LocalView(local).Data() + local_size - 1,
                               sizeof(float), cudaMemcpyDefault));
    EXPECT_FLOAT_EQ(samples[0], expected) << "GPU " << local;
    EXPECT_FLOAT_EQ(samples[1], expected) << "GPU " << local;
  }
  EXPECT_EQ(executor.TransferCount(), 0U);
}

TEST(DistributedTensorMultiGpu, DISABLED_ReplicatedTwoRankMaterialization) {
  GTEST_SKIP() << "Requires a future two-rank collective executor backend";
}
