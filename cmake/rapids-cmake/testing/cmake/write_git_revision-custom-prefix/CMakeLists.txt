# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 2.4 LANGUAGES CXX)
rapids_cmake_write_git_revision_file(demo_version
                                     "${CMAKE_CURRENT_BINARY_DIR}/demo/demo_git_version.hpp")
rapids_cmake_write_git_revision_file(
  nested_version "${CMAKE_CURRENT_BINARY_DIR}/nested/nested_git_version.h" PREFIX NESTED)

add_executable(write_git_version main.cpp)
target_link_libraries(write_git_version PRIVATE demo_version nested_version)
target_compile_features(write_git_version PRIVATE cxx_std_14)
