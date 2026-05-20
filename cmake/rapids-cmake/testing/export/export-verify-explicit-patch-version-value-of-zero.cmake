# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

project(test LANGUAGES CXX VERSION 08.06.04)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export(BUILD test VERSION 21.09.0 EXPORT_SET fake_set LANGUAGES CXX)

# Verify that build files have correct names
if(NOT EXISTS "${CMAKE_BINARY_DIR}/test-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()

# Verify that the version.cmake file exists with an explicit version arg
if(NOT EXISTS "${CMAKE_BINARY_DIR}/test-config-version.cmake")
  message(FATAL_ERROR "rapids_export failed to generate a version file")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
find_package(test 21.9 REQUIRED)
if(NOT TEST_VERSION STREQUAL "21.09.0")
  message(FATAL_ERROR "rapids_export failed to export version information")
endif()

if(NOT TEST_VERSION_MAJOR STREQUAL "21")
  message(FATAL_ERROR "rapids_export failed to export major version value")
endif()

if(NOT TEST_VERSION_MINOR STREQUAL "09")
  message(FATAL_ERROR "rapids_export failed to export minor version value")
endif()

if(NOT TEST_VERSION_PATCH STREQUAL "0")
  message(FATAL_ERROR "rapids_export failed to export patch version value")
endif()

if(NOT "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}" VERSION_EQUAL 21.09)
  message(FATAL_ERROR "rapids_export failed to export version major/minor information")
endif()

find_package(test 21.09.0 EXACT REQUIRED)
if(NOT TEST_VERSION STREQUAL "21.09.0")
  message(FATAL_ERROR "rapids_export failed to export version information")
endif()

if(NOT TEST_VERSION_MAJOR STREQUAL "21")
  message(FATAL_ERROR "rapids_export failed to export major version value")
endif()

if(NOT TEST_VERSION_MINOR STREQUAL "09")
  message(FATAL_ERROR "rapids_export failed to export minor version value")
endif()

if(NOT TEST_VERSION_PATCH STREQUAL "0")
  message(FATAL_ERROR "rapids_export failed to export patch version value")
endif()

if(NOT "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}" VERSION_EQUAL 21.09)
  message(FATAL_ERROR "rapids_export failed to export version major/minor information")
endif()
