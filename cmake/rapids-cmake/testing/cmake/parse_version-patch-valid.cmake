# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(PATCH "1.a.0" patch_value)
message(STATUS "patch_value: ${patch_value}")
if(NOT patch_value EQUAL 0)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "0.200.git-sha1" patch_value)
if(NOT patch_value STREQUAL "git-sha1")
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "21.03.00...22.04.00" patch_value)
if(NOT patch_value STREQUAL "00")
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()
