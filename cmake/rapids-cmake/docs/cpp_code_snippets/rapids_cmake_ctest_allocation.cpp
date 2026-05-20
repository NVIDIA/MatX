/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapids_cmake_ctest_allocation.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <string_view>

namespace rapids_cmake {

namespace {
GPUAllocation noGPUAllocation() { return GPUAllocation{-1, -1}; }

GPUAllocation parseCTestAllocation(std::string_view env_variable)
{
  std::string gpu_resources{std::getenv(env_variable.begin())};
  // need to handle parseCTestAllocation variable being empty

  // need to handle parseCTestAllocation variable not having some
  // of the requested components

  // The string looks like "id:<number>,slots:<number>"
  auto id_start   = gpu_resources.find("id:") + 3;
  auto id_end     = gpu_resources.find(",");
  auto slot_start = gpu_resources.find("slots:") + 6;

  auto id    = gpu_resources.substr(id_start, id_end - id_start);
  auto slots = gpu_resources.substr(slot_start);

  return GPUAllocation{std::stoi(id), std::stoi(slots)};
}

std::vector<GPUAllocation> determineGPUAllocations()
{
  std::vector<GPUAllocation> allocations;
  const auto* resource_count = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
  if (!resource_count) {
    allocations.emplace_back();
    return allocations;
  }

  const auto resource_max = std::stoi(resource_count);
  for (int index = 0; index < resource_max; ++index) {
    std::string group_env = "CTEST_RESOURCE_GROUP_" + std::to_string(index);
    std::string resource_group{std::getenv(group_env.c_str())};
    std::transform(resource_group.begin(), resource_group.end(), resource_group.begin(), ::toupper);

    if (resource_group == "GPUS") {
      auto resource_env = group_env + "_" + resource_group;
      auto&& allocation = parseCTestAllocation(resource_env);
      allocations.emplace_back(allocation);
    }
  }

  return allocations;
}
}  // namespace

bool using_resources()
{
  const auto* resource_count = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
  return resource_count != nullptr;
}

std::vector<GPUAllocation> full_allocation() { return determineGPUAllocations(); }

cudaError_t bind_to_gpu(GPUAllocation const& alloc) { return cudaSetDevice(alloc.device_id); }

bool bind_to_first_gpu()
{
  if (using_resources()) {
    std::vector<GPUAllocation> allocs = determineGPUAllocations();
    return (bind_to_gpu(allocs[0]) == cudaSuccess);
  }
  return false;
}

}  // namespace rapids_cmake
