# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/init_architectures.cmake)

set(ENV{CUDAARCHS} "80")
rapids_cuda_init_architectures(rapids-project)
project(rapids-project LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should exist after calling rapids_cuda_init_architectures()"
  )
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't ignore users CUDA_ARCHITECTURES value")
endif()
