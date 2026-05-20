# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

project(test LANGUAGES CXX VERSION 1.2)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export(BUILD test VERSION OFF EXPORT_SET fake_set LANGUAGES CXX)

# Verify that build files have correct names
if(NOT EXISTS "${CMAKE_BINARY_DIR}/test-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()

# Verify that the version.cmake file doesn't exist as we don't have a version value
if(EXISTS "${CMAKE_BINARY_DIR}/test-config-version.cmake")
  message(FATAL_ERROR "rapids_export incorrectly generated a version file")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
find_package(test REQUIRED)

if(DEFINED TEST_VERSION)
  message(FATAL_ERROR "rapids_export incorrectly generated a version variable")
endif()

if(DEFINED TEST_VERSION_MAJOR)
  message(FATAL_ERROR "rapids_export incorrectly generated a major version value")
endif()

if(DEFINED TEST_VERSION_MINOR)
  message(FATAL_ERROR "rapids_export incorrectly generated a minor version value")
endif()

if(DEFINED TEST_VERSION_PATCH)
  message(FATAL_ERROR "rapids_export incorrectly generated a patch version value")
endif()
