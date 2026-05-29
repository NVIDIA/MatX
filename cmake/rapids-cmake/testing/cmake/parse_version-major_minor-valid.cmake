# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(MAJOR_MINOR "1.a.0" major_minor)
if(NOT major_minor STREQUAL "1.a")
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR_MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR_MINOR "a_string.1" major_minor)
if(NOT major_minor STREQUAL "a_string.1")
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR_MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR_MINOR "0.200.git-sha1" major_minor)
if(NOT major_minor STREQUAL "0.200")
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR_MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MAJOR_MINOR "21.03.00...22.04.00" major_minor)
if(NOT major_minor STREQUAL "21.03")
  message(FATAL_ERROR "rapids_cmake_parse_version(MAJOR_MINOR) value parsing failed unexpectedly")
endif()
