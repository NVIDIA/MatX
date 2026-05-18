# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/init_architectures.cmake)

set(ENV{CUDAARCHS} "NATIVE")
set(ENV{CUDA_VISIBLE_DEVICES} "-1")
rapids_cuda_init_architectures(rapids-project)
project(rapids-project LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should exist after calling rapids_cuda_init_architectures()"
  )
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't init CUDA_ARCHITECTURES")
endif()

# Since we have no visible devices the output will match that of RAPIDS and not NATIVE
include("${rapids-cmake-testing-dir}/cuda/validate-cuda-rapids.cmake")
