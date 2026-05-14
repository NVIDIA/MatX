# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(MINOR "" minor_value)
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing should have failed")
endif()

rapids_cmake_parse_version(MINOR "not-a-version" minor_value)
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "100" minor_value)
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "." minor_value)
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "100." minor_value)
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "100.." minor_value)
message(STATUS "minor_value: ${minor_value}")
if(DEFINED minor_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()
