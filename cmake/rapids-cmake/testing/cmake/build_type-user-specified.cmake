# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/build_type.cmake)

set(CMAKE_BUILD_TYPE USER)

rapids_cmake_build_type(DEBUG)

if(NOT CMAKE_BUILD_TYPE STREQUAL "USER")
  message(FATAL_ERROR "rapids_cmake_build_type overwrote user setting")
endif()
