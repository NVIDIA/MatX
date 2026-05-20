# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/init_architectures.cmake)

cmake_minimum_required(VERSION 3.30.4)

unset(CMAKE_CUDA_ARCHITECTURES)
unset(ENV{CUDAARCHS})

rapids_cuda_init_architectures(rapids-project)
project(rapids-project LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't init CUDA_ARCHITECTURES")
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't init CUDA_ARCHITECTURES")
endif()

include("${rapids-cmake-testing-dir}/cuda/validate-cuda-rapids.cmake")
