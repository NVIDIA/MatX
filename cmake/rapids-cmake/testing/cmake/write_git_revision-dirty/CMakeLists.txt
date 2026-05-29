# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 2.4 LANGUAGES CXX)

rapids_cmake_write_git_revision_file(git_generated_header
                                     "${CMAKE_CURRENT_BINARY_DIR}/git_version.hpp")

execute_process(COMMAND ${GIT_EXECUTABLE} describe --tag --dirty --always
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                ERROR_QUIET
                OUTPUT_VARIABLE RAPIDS_WRITE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
)

add_executable(is_dirty main.cpp)
target_link_libraries(is_dirty PRIVATE git_generated_header)
target_compile_features(is_dirty PRIVATE cxx_std_14)

if(RAPIDS_WRITE_VERSION MATCHES "dirty")
  target_compile_definitions(is_dirty PRIVATE "-DIS_DIRTY")
endif()
