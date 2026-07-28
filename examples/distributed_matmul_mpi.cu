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

index_t ParseDimension(const char *text, const char *name) {
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

float DiagonalValue(index_t row) {
  return static_cast<float>(row % 7 + 1);
}

float BValue(index_t row, index_t column) {
  return 0.125F +
         0.25F * static_cast<float>((row * 3 + column * 5) % 19);
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

template <typename Tensor, typename Expected>
float LocalMaximumError(const Tensor &tensor, Expected &&expected) {
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
      const float actual =
          host[static_cast<size_t>(row * shape[1] + column)];
      maximum_error =
          std::max(maximum_error,
                   std::abs(actual - expected(global[0], global[1])));
    }
  }
  return maximum_error;
}

int RunExample(int argc, char **argv, int world_rank) {
  int world_size = 0;
  CheckMpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");

  const index_t m = argc > 1 ? ParseDimension(argv[1], "M") : 1024;
  const index_t k = argc > 2 ? ParseDimension(argv[2], "K") : 1024;
  const index_t n = argc > 3 ? ParseDimension(argv[3], "N") : 1024;
  const index_t block = argc > 4 ? ParseDimension(argv[4], "block") : 128;
  if (argc > 5) {
    if (world_rank == 0) {
      std::cerr << "Usage: distributed_matmul_mpi [M [K [N [block]]]]\n";
    }
    return 2;
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
  // Launchers commonly either expose every node-local GPU to every rank or
  // constrain each rank to one distinct physical GPU with CUDA_VISIBLE_DEVICES.
  const int local_device_error =
      visible_devices == 1 || visible_devices >= local_size ? 0 : 1;
  int any_device_error = 0;
  CheckMpi(MPI_Allreduce(&local_device_error, &any_device_error, 1, MPI_INT,
                         MPI_MAX, MPI_COMM_WORLD),
           "MPI_Allreduce(device availability)");
  if (any_device_error != 0) {
    if (world_rank == 0) {
      std::cerr << "Each node must expose at least one CUDA GPU per local MPI "
                   "rank, either collectively or through per-rank visibility\n";
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

  block_cyclic_distribution_t a_distribution{
      {m, k}, {block, block}, process_grid, endpoints};
  block_cyclic_distribution_t b_distribution{
      {k, n}, {block, block}, process_grid, endpoints};
  block_cyclic_distribution_t c_distribution{
      {m, n}, {block, block}, process_grid, endpoints};
  auto a = make_distributed_tensor<float>(a_distribution, context);
  auto b = make_distributed_tensor<float>(b_distribution, context);
  auto c = make_distributed_tensor<float>(c_distribution, context);

  FillLocalMatrix(a, [](index_t row, index_t column) {
    return row == column ? DiagonalValue(row) : 0.0F;
  });
  FillLocalMatrix(
      b, [](index_t row, index_t column) { return BValue(row, column); });
  FillLocalMatrix(c, [](index_t, index_t) { return 0.0F; });

  // Every rank enters this expression collectively. MatX dispatches this
  // block-cyclic operation to cuBLASMp through the distributed executor.
  (c = matmul(a, b)).run(executor);
  executor.sync();

  const float local_error = LocalMaximumError(c, [k](index_t row,
                                                     index_t column) {
    return row < k ? DiagonalValue(row) * BValue(row, column) : 0.0F;
  });
  float maximum_error = 0.0F;
  CheckMpi(MPI_Allreduce(&local_error, &maximum_error, 1, MPI_FLOAT, MPI_MAX,
                         MPI_COMM_WORLD),
           "MPI_Allreduce(maximum error)");

  const int result = maximum_error <= 1.0e-4F ? 0 : 1;
  if (world_rank == 0) {
    std::cout << "cuBLASMp GEMM " << m << "x" << k << " * " << k << "x" << n
              << " across " << world_size << " ranks in a " << process_grid[0]
              << "x" << process_grid[1]
              << " process grid; maximum error " << maximum_error << '\n';
  }
  CheckMpi(MPI_Comm_free(&local_comm), "MPI_Comm_free");
  return result;
}

} // namespace

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
      MPI_SUCCESS) {
    std::cerr << "MPI_Init_thread failed\n";
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

  (void)MPI_Finalize();
  return result;
}
