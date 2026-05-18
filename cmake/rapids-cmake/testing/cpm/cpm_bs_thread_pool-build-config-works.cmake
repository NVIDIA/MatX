# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

rapids_cpm_init()
rapids_cpm_bs_thread_pool(BUILD_EXPORT_SET test)

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir/CMakeLists.txt"
     "
cmake_minimum_required(VERSION 3.30.4)
project(verify_bs_thread_pool LANGUAGES CXX)

set(CMAKE_PREFIX_PATH \"${CMAKE_BINARY_DIR}/\")
find_package(bs_thread_pool REQUIRED)

file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/stub.cpp\" \"#include <BS_thread_pool.hpp>\")
add_library(uses_bs_thread_pool SHARED stub.cpp)
target_link_libraries(uses_bs_thread_pool PRIVATE BS::thread_pool)
")

add_custom_target(verify_build_config ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf
                          "${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir"
                          -B="${CMAKE_BINARY_DIR}/build"
                  COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/build")
