# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/init_architectures.cmake)

cmake_minimum_required(VERSION 3.23.1)

set(CMAKE_CUDA_ARCHITECTURES 80-real)
set(ENV{CUDAARCHS} "9000")

rapids_cuda_init_architectures(rapids-project)
project(rapids-project LANGUAGES CUDA)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "9000")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't init CUDA_ARCHITECTURES")
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80-real")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't preserve users CUDA_ARCHITECTURES value"
  )
endif()
