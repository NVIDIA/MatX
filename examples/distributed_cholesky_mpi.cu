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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <nccl.h>

using namespace matx;
using namespace matx::experimental;

namespace {

void CheckNccl(ncclResult_t status, const char *operation) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string{operation} + ": " +
                             ncclGetErrorString(status));
  }
}

void CheckMpi(int status, const char *operation) {
  if (status == MPI_SUCCESS) {
    return;
  }

  char error[MPI_MAX_ERROR_STRING]{};
  int length = 0;
  (void)MPI_Error_string(status, error, &length);
  throw std::runtime_error(std::string{operation} + ": " +
                           std::string{error, static_cast<size_t>(length)});
}

index_t ParsePositive(const char *text, const char *name) {
  size_t parsed = 0;
  const long long value = std::stoll(text, &parsed);
  if (text[parsed] != '\0' || value <= 0) {
    throw std::invalid_argument(std::string{name} +
                                " must be a positive integer");
  }
  return static_cast<index_t>(value);
}

class NcclCommunicator {
public:
  NcclCommunicator(int rank, int size) {
    ncclUniqueId id{};
    if (rank == 0) {
      CheckNccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    CheckMpi(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
             "MPI_Bcast(NCCL ID)");
    CheckNccl(ncclCommInitRank(&communicator_, size, id, rank),
              "ncclCommInitRank");
  }

  NcclCommunicator(const NcclCommunicator &) = delete;
  NcclCommunicator &operator=(const NcclCommunicator &) = delete;

  ~NcclCommunicator() {
    if (communicator_ != nullptr) {
      (void)ncclCommDestroy(communicator_);
    }
  }

  ncclComm_t get() const noexcept { return communicator_; }

private:
  ncclComm_t communicator_ = nullptr;
};

distributed_index_t<2> ProcessGrid(int process_count) {
  int rows = static_cast<int>(std::sqrt(static_cast<double>(process_count)));
  while (process_count % rows != 0) {
    --rows;
  }
  return {static_cast<index_t>(rows),
          static_cast<index_t>(process_count / rows)};
}

float CholeskyDiagonal(index_t row) {
  return static_cast<float>(row % 17 + 2);
}

template <typename Tensor, typename Generator>
void FillLocalMatrix(Tensor &tensor, Generator &&generator) {
  if (tensor.LocalFragmentCount() != 1) {
    throw std::runtime_error("Expected exactly one local matrix fragment");
  }

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

template <typename Tensor>
float LocalMaximumError(const Tensor &tensor) {
  const auto &fragment = tensor.LocalFragment(0);
  const auto &distribution = tensor.DistributionDescriptor();
  const auto shape = distribution.LocalShape(fragment.distribution_index);
  std::vector<float> host(static_cast<size_t>(shape[0] * shape[1]));
  MATX_CUDA_CHECK(cudaMemcpy(host.data(), tensor.LocalView(0).Data(),
                             host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

  float maximum_error = 0.0F;
  for (index_t row = 0; row < shape[0]; ++row) {
    for (index_t column = 0; column < shape[1]; ++column) {
      const auto global = distribution.LocalToGlobal(
          fragment.distribution_index, {row, column});
      if (global[0] < global[1]) {
        continue;
      }
      const float expected =
          global[0] == global[1] ? CholeskyDiagonal(global[0]) : 0.0F;
      const float actual =
          host[static_cast<size_t>(row * shape[1] + column)];
      maximum_error =
          std::max(maximum_error, std::abs(actual - expected));
    }
  }
  return maximum_error;
}

int RunExample(int argc, char **argv, int world_rank) {
  int world_size = 0;
  CheckMpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
  if (world_size < 2) {
    if (world_rank == 0) {
      std::cerr << "Launch at least two MPI ranks; each rank uses one GPU\n";
    }
    return 2;
  }
  if (argc > 3) {
    if (world_rank == 0) {
      std::cerr << "Usage: distributed_cholesky_mpi [matrix_size "
                   "[block_size]]\n";
    }
    return 2;
  }

  const index_t matrix_size =
      argc > 1 ? ParsePositive(argv[1], "matrix_size") : 1024;
  const index_t block_size =
      argc > 2 ? ParsePositive(argv[2], "block_size") : 128;
  if (block_size > matrix_size) {
    throw std::invalid_argument("block_size cannot exceed matrix_size");
  }

  MPI_Comm local_comm = MPI_COMM_NULL;
  CheckMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                               MPI_INFO_NULL, &local_comm),
           "MPI_Comm_split_type");

  int local_rank = 0;
  int local_size = 0;
  CheckMpi(MPI_Comm_rank(local_comm, &local_rank), "MPI_Comm_rank(local)");
  CheckMpi(MPI_Comm_size(local_comm, &local_size), "MPI_Comm_size(local)");

  int visible_devices = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&visible_devices));
  // A launcher may expose every node-local GPU to every rank, or expose one
  // distinct physical GPU as device zero to each rank.
  const int local_device_error =
      visible_devices == 1 || visible_devices >= local_size ? 0 : 1;
  int any_device_error = 0;
  CheckMpi(MPI_Allreduce(&local_device_error, &any_device_error, 1, MPI_INT,
                         MPI_MAX, MPI_COMM_WORLD),
           "MPI_Allreduce(device availability)");
  if (any_device_error != 0) {
    if (world_rank == 0) {
      std::cerr << "Each node must expose at least one GPU per local MPI rank\n";
    }
    CheckMpi(MPI_Comm_free(&local_comm), "MPI_Comm_free");
    return 2;
  }

  const int device = visible_devices == 1 ? 0 : local_rank;
  MATX_CUDA_CHECK(cudaSetDevice(device));

  std::vector<int> rank_devices(static_cast<size_t>(world_size));
  CheckMpi(MPI_Allgather(&device, 1, MPI_INT, rank_devices.data(), 1, MPI_INT,
                         MPI_COMM_WORLD),
           "MPI_Allgather(rank devices)");

  std::vector<distributed_endpoint_t> endpoints;
  endpoints.reserve(static_cast<size_t>(world_size));
  for (int rank = 0; rank < world_size; ++rank) {
    endpoints.push_back({rank, rank_devices[static_cast<size_t>(rank)]});
  }

  const auto process_grid = ProcessGrid(world_size);
  NcclCommunicator communicator{world_rank, world_size};
  distributed_context context{{device}, world_rank, world_size};
  distributedCUDAExecutor executor{
      context, communicator.get(), static_cast<int>(process_grid[0]),
      static_cast<int>(process_grid[1])};

  block_cyclic_distribution_t distribution{
      {matrix_size, matrix_size},
      {block_size, block_size},
      process_grid,
      endpoints};
  auto input = make_distributed_tensor<float>(distribution, context);
  auto output = make_distributed_tensor<float>(distribution, context);

  // A positive diagonal matrix makes the expected lower factor unambiguous.
  FillLocalMatrix(input, [](index_t row, index_t column) {
    if (row != column) {
      return 0.0F;
    }
    const float diagonal = CholeskyDiagonal(row);
    return diagonal * diagonal;
  });

  // Every MPI rank enters this expression collectively. The rank count,
  // selected with mpirun -n, is also the number of GPUs used.
  (output = chol(input, SolverFillMode::LOWER)).run(executor);
  executor.sync();

  const float local_error = LocalMaximumError(output);
  float maximum_error = 0.0F;
  CheckMpi(MPI_Allreduce(&local_error, &maximum_error, 1, MPI_FLOAT, MPI_MAX,
                         MPI_COMM_WORLD),
           "MPI_Allreduce(maximum error)");

  if (world_rank == 0) {
    std::cout << "cuSOLVERMp Cholesky of a " << matrix_size << "x"
              << matrix_size << " matrix across " << world_size
              << " MPI ranks/GPUs in a " << process_grid[0] << "x"
              << process_grid[1] << " process grid; maximum error "
              << maximum_error << '\n';
  }

  CheckMpi(MPI_Comm_free(&local_comm), "MPI_Comm_free");
  return maximum_error <= 1.0e-4F ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed\n";
    return 1;
  }

  int world_rank = -1;
  (void)MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int result = 1;
  try {
    result = RunExample(argc, argv, world_rank);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Rank %d failed: %s\n", world_rank, error.what());
    (void)MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ClearCachesAndAllocations();
  (void)MPI_Finalize();
  return result;
}
