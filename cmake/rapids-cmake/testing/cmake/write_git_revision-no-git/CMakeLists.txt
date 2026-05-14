# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 2.4 LANGUAGES CXX)

add_subdirectory(gen_header)

add_executable(verify_no_git main.cpp)
target_link_libraries(verify_no_git PRIVATE git_generated_header)
target_compile_features(verify_no_git PRIVATE cxx_std_14)
