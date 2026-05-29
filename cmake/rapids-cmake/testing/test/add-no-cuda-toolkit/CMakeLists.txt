# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/rapids-test.cmake)

# Disable searching for the CUDA Toolkit so we emulate support for CPU only rapids-cmake-testing
# support
set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit True)

rapids_test_init()

add_executable(verify_alloc main.cpp)
enable_testing()
rapids_test_add(NAME simple_test COMMAND verify_alloc)
