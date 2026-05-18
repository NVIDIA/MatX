# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/set_architectures.cmake)

set(user_value "user-value")
set(CMAKE_CUDA_ARCHITECTURES ${user_value})
rapids_cuda_set_architectures(invalid-mode)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL user_value)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES shouldn't be modified by "
                      "rapids_cuda_set_architectures() when past an invalid mode")
endif()
