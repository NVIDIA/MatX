////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include "matx/distributed.h"
#include "gtest/gtest.h"

#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace matx;
using namespace matx::experimental;

namespace {

class NcclCommunicator {
public:
  NcclCommunicator(int rank, int size) {
    ncclUniqueId id{};
    if (rank == 0) {
      const ncclResult_t status = ncclGetUniqueId(&id);
      if (status != ncclSuccess) {
        throw std::runtime_error("ncclGetUniqueId failed");
      }
    }
    if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) !=
        MPI_SUCCESS) {
      throw std::runtime_error("MPI_Bcast failed");
    }
    const ncclResult_t status =
        ncclCommInitRank(&communicator_, size, id, rank);
    if (status != ncclSuccess) {
      throw std::runtime_error("ncclCommInitRank failed");
    }
  }

  NcclCommunicator(const NcclCommunicator &) = delete;
  NcclCommunicator &operator=(const NcclCommunicator &) = delete;

  ~NcclCommunicator() {
    if (communicator_ != nullptr) {
      (void)ncclCommDestroy(communicator_);
    }
  }

  [[maybe_unused]] ncclComm_t get() const noexcept { return communicator_; }

private:
  ncclComm_t communicator_ = nullptr;
};

struct MpEnvironment {
  int rank = 0;
  int size = 0;
  int device = 0;
  int local_size = 0;
  int process_rows = 1;
  int process_columns = 1;
  std::vector<distributed_endpoint_t> endpoints;
};

[[maybe_unused]] MpEnvironment GetMpEnvironment() {
  MpEnvironment environment;
  EXPECT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &environment.rank), MPI_SUCCESS);
  EXPECT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &environment.size), MPI_SUCCESS);

  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  MPI_Comm local_communicator = MPI_COMM_NULL;
  EXPECT_EQ(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                environment.rank, MPI_INFO_NULL,
                                &local_communicator),
            MPI_SUCCESS);
  int local_rank = 0;
  EXPECT_EQ(MPI_Comm_rank(local_communicator, &local_rank), MPI_SUCCESS);
  EXPECT_EQ(MPI_Comm_size(local_communicator, &environment.local_size),
            MPI_SUCCESS);
  environment.device = local_rank;
  EXPECT_EQ(MPI_Comm_free(&local_communicator), MPI_SUCCESS);

  environment.process_rows = environment.size % 2 == 0 ? 2 : 1;
  environment.process_columns = environment.size / environment.process_rows;
  std::vector<int> devices(static_cast<size_t>(environment.size));
  EXPECT_EQ(MPI_Allgather(&environment.device, 1, MPI_INT, devices.data(), 1,
                          MPI_INT, MPI_COMM_WORLD),
            MPI_SUCCESS);
  environment.endpoints.reserve(devices.size());
  for (int rank = 0; rank < environment.size; ++rank) {
    environment.endpoints.push_back({rank, devices[static_cast<size_t>(rank)]});
  }
  return environment;
}

template <typename Tensor, typename Generator>
void FillLocalMatrix(Tensor &tensor, Generator &&generator) {
  ASSERT_EQ(tensor.LocalFragmentCount(), 1U);
  const auto &fragment = tensor.LocalFragment(0);
  const auto &distribution = tensor.DistributionDescriptor();
  const auto shape = distribution.LocalShape(fragment.distribution_index);
  std::vector<float> host(static_cast<size_t>(shape[0] * shape[1]));
  for (index_t row = 0; row < shape[0]; ++row) {
    for (index_t column = 0; column < shape[1]; ++column) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {row, column});
      host[static_cast<size_t>(row * shape[1] + column)] =
          generator(global[0], global[1]);
    }
  }
  MATX_CUDA_CHECK(cudaMemcpy(tensor.LocalView(0).Data(), host.data(),
                             host.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
}

template <typename Tensor, typename Verifier>
void VerifyLocalMatrix(const Tensor &tensor, Verifier &&verifier) {
  ASSERT_EQ(tensor.LocalFragmentCount(), 1U);
  const auto &fragment = tensor.LocalFragment(0);
  const auto &distribution = tensor.DistributionDescriptor();
  const auto shape = distribution.LocalShape(fragment.distribution_index);
  std::vector<float> host(static_cast<size_t>(shape[0] * shape[1]));
  MATX_CUDA_CHECK(cudaMemcpy(host.data(), tensor.LocalView(0).Data(),
                             host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
  for (index_t row = 0; row < shape[0]; ++row) {
    for (index_t column = 0; column < shape[1]; ++column) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {row, column});
      verifier(global[0], global[1],
               host[static_cast<size_t>(row * shape[1] + column)]);
    }
  }
}

} // namespace

#ifdef MATX_EN_NCCL
TEST(DistributedMpIntegration, NcclDistributedSum) {
  int rank = 0;
  int size = 0;
  ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  MPI_Comm local_communicator = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_communicator),
            MPI_SUCCESS);
  int local_rank = 0;
  int local_size = 0;
  ASSERT_EQ(MPI_Comm_rank(local_communicator, &local_rank), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_size(local_communicator, &local_size), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_free(&local_communicator), MPI_SUCCESS);

  int gpus_per_process = 1;
  if (const char *configured = std::getenv("MATX_MNMG_GPUS_PER_PROCESS")) {
    gpus_per_process = std::max(1, std::atoi(configured));
  }
  if (size < 2 || device_count < local_size * gpus_per_process) {
    GTEST_SKIP() << "Requires two MPI ranks and " << gpus_per_process
                 << " non-overlapping GPU(s) per process";
  }

  std::vector<int> local_devices(static_cast<size_t>(gpus_per_process));
  for (int local = 0; local < gpus_per_process; ++local)
    local_devices[static_cast<size_t>(local)] =
        local_rank * gpus_per_process + local;
  std::vector<int> all_devices(static_cast<size_t>(size * gpus_per_process));
  ASSERT_EQ(MPI_Allgather(local_devices.data(), gpus_per_process, MPI_INT,
                          all_devices.data(), gpus_per_process, MPI_INT,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  std::vector<distributed_endpoint_t> endpoints;
  endpoints.reserve(all_devices.size());
  for (int process = 0; process < size; ++process)
    for (int local = 0; local < gpus_per_process; ++local)
      endpoints.push_back({process, all_devices[static_cast<size_t>(
                                        process * gpus_per_process + local)]});

  ncclUniqueId id{};
  if (rank == 0)
    matx::detail::NcclCheck(ncclGetUniqueId(&id), "ncclGetUniqueId");
  ASSERT_EQ(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
            MPI_SUCCESS);
  std::vector<ncclComm_t> communicators(static_cast<size_t>(gpus_per_process),
                                        nullptr);
  matx::detail::NcclCheck(ncclGroupStart(), "ncclGroupStart");
  for (int local = 0; local < gpus_per_process; ++local) {
    MATX_CUDA_CHECK(cudaSetDevice(local_devices[static_cast<size_t>(local)]));
    matx::detail::NcclCheck(
        ncclCommInitRank(&communicators[static_cast<size_t>(local)],
                         size * gpus_per_process, id,
                         rank * gpus_per_process + local),
        "ncclCommInitRank");
  }
  matx::detail::NcclCheck(ncclGroupEnd(), "ncclGroupEnd");

  {
    distributed_context context{local_devices, rank, size};
    std::vector<distributed_nccl_binding_t> bindings;
    for (int local = 0; local < gpus_per_process; ++local)
      bindings.push_back({{rank, local_devices[static_cast<size_t>(local)]},
                          communicators[static_cast<size_t>(local)]});
    distributed_nccl_topology_t topology{endpoints, std::move(bindings)};
    distributedCUDAExecutor executor{context, std::move(topology)};

    constexpr index_t count = 37;
    auto distribution = block_distribution_t<1>::Slab({count}, endpoints);
    auto input = make_distributed_tensor<float>(distribution, context);
    for (size_t local = 0; local < input.LocalFragmentCount(); ++local) {
      const auto &fragment = input.LocalFragment(local);
      const auto shape = distribution.LocalShape(fragment.distribution_index);
      std::vector<float> host(static_cast<size_t>(shape[0]));
      for (index_t i = 0; i < shape[0]; ++i) {
        const auto global =
            distribution.LocalToGlobal(fragment.distribution_index, {i});
        host[static_cast<size_t>(i)] = static_cast<float>(global[0] + 1);
      }
      matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
      MATX_CUDA_CHECK(cudaMemcpy(input.LocalView(local).Data(), host.data(),
                                 host.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));
    }

    replicated_distribution_t<0> output_distribution{distributed_index_t<0>{},
                                                     endpoints};
    auto output = make_distributed_tensor<float>(output_distribution, context);
    (output = sum(input)).run(executor);
    executor.sync();
    for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
      const auto &fragment = output.LocalFragment(local);
      matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
      float result = 0.0F;
      MATX_CUDA_CHECK(cudaMemcpy(&result, output.LocalView(local).Data(),
                                 sizeof(float), cudaMemcpyDeviceToHost));
      EXPECT_FLOAT_EQ(result, 703.0F);
    }
  }
  for (auto communicator : communicators)
    if (communicator != nullptr)
      (void)ncclCommDestroy(communicator);
}
#endif

#ifdef MATX_EN_CUBLASMP
TEST(DistributedMpIntegration, CublasMpMatmul) {
  const auto environment = GetMpEnvironment();
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (environment.size < 2 || device_count < environment.local_size) {
    GTEST_SKIP() << "Requires at least two MPI ranks and one GPU per rank";
  }

  MATX_CUDA_CHECK(cudaSetDevice(environment.device));
  NcclCommunicator communicator{environment.rank, environment.size};
  distributed_context context{
      {environment.device}, environment.rank, environment.size};
  distributedCUDAExecutor executor{context, communicator.get(),
                                   environment.process_rows,
                                   environment.process_columns};

  constexpr index_t matrix_size = 16;
  block_cyclic_distribution_t distribution{
      {matrix_size, matrix_size},
      {2, 2},
      {environment.process_rows, environment.process_columns},
      environment.endpoints};
  auto a = make_distributed_tensor<float>(distribution, context);
  auto b = make_distributed_tensor<float>(distribution, context);
  auto c = make_distributed_tensor<float>(distribution, context);

  FillLocalMatrix(a, [](index_t row, index_t column) {
    return row == column ? static_cast<float>(row + 1) : 0.0F;
  });
  FillLocalMatrix(b, [](index_t row, index_t column) {
    return static_cast<float>(row * matrix_size + column + 1);
  });
  FillLocalMatrix(c, [](index_t, index_t) { return 0.0F; });

  (c = matmul(a, b)).run(executor);
  executor.sync();

  VerifyLocalMatrix(c, [](index_t row, index_t column, float value) {
    const float expected =
        static_cast<float>((row + 1) * (row * matrix_size + column + 1));
    EXPECT_NEAR(value, expected, 2.0e-3F);
  });
}
#endif

#ifdef MATX_EN_CUSOLVERMP
TEST(DistributedMpIntegration, CusolverMpCholesky) {
  const auto environment = GetMpEnvironment();
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (environment.size < 2 || device_count < environment.local_size) {
    GTEST_SKIP() << "Requires at least two MPI ranks and one GPU per rank";
  }

  MATX_CUDA_CHECK(cudaSetDevice(environment.device));
  NcclCommunicator communicator{environment.rank, environment.size};
  distributed_context context{
      {environment.device}, environment.rank, environment.size};
  distributedCUDAExecutor executor{context, communicator.get(),
                                   environment.process_rows,
                                   environment.process_columns};

  constexpr index_t matrix_size = 16;
  block_cyclic_distribution_t distribution{
      {matrix_size, matrix_size},
      {2, 2},
      {environment.process_rows, environment.process_columns},
      environment.endpoints};
  auto a = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  FillLocalMatrix(a, [](index_t row, index_t column) {
    const float diagonal = static_cast<float>(row + 2);
    return row == column ? diagonal * diagonal : 0.0F;
  });

  (output = chol(a, SolverFillMode::LOWER)).run(executor);
  executor.sync();

  VerifyLocalMatrix(output, [](index_t row, index_t column, float value) {
    if (row >= column) {
      const float expected = row == column ? static_cast<float>(row + 2) : 0.0F;
      EXPECT_NEAR(value, expected, 2.0e-3F);
    }
  });
}
#endif

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  matx::ClearCachesAndAllocations();
  MPI_Finalize();
  return result;
}
