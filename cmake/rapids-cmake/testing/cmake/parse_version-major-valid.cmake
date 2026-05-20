# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(MAJOR "not-a-version" major_value)
if(NOT major_value STREQUAL "not-a-version")
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR "100" major_value)
if(NOT major_value EQUAL 100)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR "1.0.a.0" major_value)
if(NOT major_value EQUAL 1)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR "0.200.git-sha1" major_value)
if(NOT major_value EQUAL 0)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR "21.03.00...22.04.00" major_value)
if(NOT major_value EQUAL 21)
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR) value parsing failed unexpectedly")
endif()
