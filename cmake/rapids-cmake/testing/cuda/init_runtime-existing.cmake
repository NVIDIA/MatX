# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/init_runtime.cmake)

set(user_value "fake-value")
set(CMAKE_CUDA_RUNTIME_LIBRARY ${user_value})

rapids_cuda_init_runtime(USE_STATIC TRUE)
if(NOT CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL user_value)
  message(FATAL_ERROR "rapids_cuda_init_runtime shouldn't override user value")
endif()

rapids_cuda_init_runtime(USE_STATIC FALSE)
if(NOT CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL user_value)
  message(FATAL_ERROR "rapids_cuda_init_runtime shouldn't override user value")
endif()
