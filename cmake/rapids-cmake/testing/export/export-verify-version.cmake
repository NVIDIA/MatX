# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

project(test LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export(BUILD test EXPORT_SET fake_set LANGUAGES CXX)

# Verify that build files have correct names
if(NOT EXISTS "${CMAKE_BINARY_DIR}/test-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()

unset(TEST_VERSION)
unset(TEST_VERSION_MAJOR)
unset(TEST_VERSION_MINOR)

set(CMAKE_FIND_PACKAGE_NAME test) # Emulate `find_package`
include("${CMAKE_BINARY_DIR}/test-config.cmake")

if(NOT TEST_VERSION VERSION_EQUAL 3.1.4)
  message(FATAL_ERROR "rapids_export failed to export version information")
endif()

if(NOT "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}" VERSION_EQUAL 3.1)
  message(FATAL_ERROR "rapids_export failed to export version major/minor information")
endif()
