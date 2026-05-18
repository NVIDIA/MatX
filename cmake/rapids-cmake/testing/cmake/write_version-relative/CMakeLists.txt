# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_version_file.cmake)
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 3.2.0 LANGUAGES NONE)
rapids_cmake_write_version_file(demo_version.hpp)

project(NESTED VERSION 3.14.159 LANGUAGES NONE)
rapids_cmake_write_version_file(nested_version.hpp)

enable_language(CXX)
add_executable(write_version main.cpp)
target_include_directories(write_version PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(write_version PRIVATE cxx_std_14)

check_copyright_header("${CMAKE_CURRENT_BINARY_DIR}/demo_version.hpp")
check_copyright_header("${CMAKE_CURRENT_BINARY_DIR}/nested_version.hpp")
