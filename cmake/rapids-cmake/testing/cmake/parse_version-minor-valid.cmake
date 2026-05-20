# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(MINOR "1.a.0" minor_value)
if(NOT minor_value STREQUAL "a")
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "0.200.git-sha1" minor_value)
if(NOT minor_value EQUAL 200)
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(MINOR "21.03.00...22.04.00" minor_value)
if(NOT minor_value STREQUAL "03")
  message(FATAL_ERROR "rapids_cmake_parse_version(MINOR) value parsing failed unexpectedly")
endif()
