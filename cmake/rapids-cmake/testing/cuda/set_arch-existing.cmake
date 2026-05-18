# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/set_architectures.cmake)

# Required by `NATIVE` as it does compiler detection
enable_language(CUDA)

set(user_value "user-value")
set(CMAKE_CUDA_ARCHITECTURES ${user_value})
rapids_cuda_set_architectures(RAPIDS)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should exist after calling rapids_cuda_set_architectures()"
  )
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL user_value)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should be modified by calling rapids_cuda_set_architectures()"
  )
endif()

set(CMAKE_CUDA_ARCHITECTURES ${user_value})
rapids_cuda_set_architectures(NATIVE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should exist after calling rapids_cuda_set_architectures()"
  )
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL user_value)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should be modified by calling rapids_cuda_set_architectures()"
  )
endif()
