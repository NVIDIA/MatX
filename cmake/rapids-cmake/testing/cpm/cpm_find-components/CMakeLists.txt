# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

rapids_cpm_init()
rapids_cpm_find(FakeDependency 11 REQUIRED COMPONENTS A B C INSTALL_EXPORT_SET test_export_set
                BUILD_EXPORT_SET test_export_set)

if(NOT (TARGET FakeDependency::A AND TARGET FakeDependency::B AND TARGET FakeDependency::C))
  message(FATAL_ERROR "rapids_cpm_find() failed to propagate COMPONENT A B C")
endif()

set(to_match_string "COMPONENTS A B C")

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FakeDependency.cmake")
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_find(BUILD) failed to preserve version information in exported file"
  )
endif()

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/package_FakeDependency.cmake")
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_find(INSTALL) failed to preserve version information in exported file"
  )
endif()
