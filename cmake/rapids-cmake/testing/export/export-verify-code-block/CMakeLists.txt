# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/export.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

set(code_block
    [=[
set(code_block_variable "Unique String To Verify")
if(NOT DEFINED rapids_namespaced_global_targets)
  message(FATAL_ERROR "rapids_export failed giving global targets to FINAL_CODE_BLOCK")
endif()
]=])

rapids_export(BUILD FakEProJecT
              EXPORT_SET fake_set
              NAMESPACE test::
              GLOBAL_TARGETS fakeLib
              FINAL_CODE_BLOCK code_block)

# Add a custom command that generates a second project that verifies our target was exported with
# the correct namespace
file(WRITE "${CMAKE_BINARY_DIR}/verify/CMakeLists.txt"
     [=[
cmake_minimum_required(VERSION 3.30.4)
project(verify_build_targets LANGUAGES CXX)
message(STATUS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-config.cmake")
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()
set(CMAKE_FIND_PACKAGE_NAME FakEProJecT) # Emulate `find_package`
include("${CMAKE_CURRENT_LIST_DIR}/../fakeproject-config.cmake")

if(NOT code_block_variable STREQUAL "Unique String To Verify")
  message(FATAL_ERROR "rapids_export failed to generate correct final code block")
endif()
  ]=])

add_custom_target(verify_target_file ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/verify/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/verify"
                          -B="${CMAKE_BINARY_DIR}/verify/build")
