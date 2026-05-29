# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(PATCH "" patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing should have failed")
endif()

rapids_cmake_parse_version(PATCH "not-a-version" patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "100" patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "." patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "21.03" patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "100.." patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH ".." patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "100..." patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "..." patch_value)
if(DEFINED patch_value)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()
