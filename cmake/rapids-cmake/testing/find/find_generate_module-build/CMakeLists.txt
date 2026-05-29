# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/generate_module.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Generate the find module
rapids_find_generate_module(RapidsTest
                            HEADER_NAMES rapids-cmake-test-header.h
                            LIBRARY_NAMES rapids-cmake-test
                            VERSION 42
                            INCLUDE_SUFFIXES rapids-cmake
                            BUILD_EXPORT_SET test_set)

find_package(RapidsTest REQUIRED)

if(NOT RapidsTest_FOUND)
  message(FATAL_ERROR "RapidsTest_FOUND should be set to TRUE")
endif()
if(NOT RAPIDSTEST_FOUND)
  message(FATAL_ERROR "RAPIDSTEST_FOUND should be set to TRUE")
endif()

if(NOT TARGET RapidsTest::RapidsTest)
  message(FATAL_ERROR "RapidsTest target should be generated")
endif()

get_target_property(include_dir RapidsTest::RapidsTest INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(release_lib RapidsTest::RapidsTest IMPORTED_LOCATION_RELEASE)
get_target_property(debug_lib RapidsTest::RapidsTest IMPORTED_LOCATION_DEBUG)

if(NOT IS_DIRECTORY "${include_dir}")
  message(FATAL_ERROR "rapids_find_generate_module failed to generate correct include dir search logic"
  )
endif()

if(NOT EXISTS "${release_lib}")
  message(FATAL_ERROR "rapids_find_generate_module failed to generate correct library search logic")
endif()

if(NOT EXISTS "${debug_lib}")
  message(FATAL_ERROR "rapids_find_generate_module failed to generate correct debug/versioned library search logic"
  )
endif()
