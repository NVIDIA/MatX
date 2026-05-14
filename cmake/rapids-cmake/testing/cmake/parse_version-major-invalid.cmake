# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(MAJOR "" major_value)

if(DEFINED major_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing should have failed")
endif()

rapids_cmake_parse_version(MAJOR "." major_value)
if(DEFINED major_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()
