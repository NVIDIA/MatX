# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

rapids_cpm_init()

if(TARGET nvtx3-c)
  message(FATAL_ERROR "Expected nvtx3-c not to exist")
endif()

if(TARGET nvtx3-cpp)
  message(FATAL_ERROR "Expected nvtx3-cpp not to exist")
endif()

rapids_cpm_nvtx3()

if(NOT TARGET nvtx3-c)
  message(FATAL_ERROR "Expected nvtx3-c target to exist")
endif()

if(NOT TARGET nvtx3-cpp)
  message(FATAL_ERROR "Expected nvtx3-cpp target to exist")
endif()

if(NOT TARGET nvtx3::nvtx3-c)
  message(FATAL_ERROR "Expected nvtx3::nvtx3-c target to exist")
endif()

if(NOT TARGET nvtx3::nvtx3-cpp)
  message(FATAL_ERROR "Expected nvtx3::nvtx3-cpp target to exist")
endif()

rapids_cpm_nvtx3()
