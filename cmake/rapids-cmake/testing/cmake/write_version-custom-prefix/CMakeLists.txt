# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_version_file.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(RandomProjectName VERSION 3.2.0 LANGUAGES NONE)
rapids_cmake_write_version_file(demo_version.hpp PREFIX DEMO)
rapids_cmake_write_version_file(nested_version.hpp PREFIX NESTED)

enable_language(CXX)
add_executable(write_version main.cpp)
target_include_directories(write_version PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(write_version PRIVATE cxx_std_14)
