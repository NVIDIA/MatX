# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/generate_module.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Generate the find module
set(initial_code_block [[set(RapidsTest_INITIAL_CODE_BLOCK TRUE)]])
set(final_code_block [[set(RapidsTest_FINAL_CODE_BLOCK TRUE)]])
rapids_find_generate_module(RapidsTest
                            HEADER_NAMES rapids-cmake-test-header_only.hpp INITIAL_CODE_BLOCK
                                         initial_code_block FINAL_CODE_BLOCK final_code_block
                            INSTALL_EXPORT_SET test_set)

find_package(RapidsTest REQUIRED)

if(NOT RapidsTest_INITIAL_CODE_BLOCK)
  message(FATAL_ERROR "RapidsTest_FOUND should be set to TRUE")
endif()
if(NOT RapidsTest_FINAL_CODE_BLOCK)
  message(FATAL_ERROR "RAPIDSTEST_FOUND should be set to TRUE")
endif()
