# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

rapids_cpm_init()
rapids_cpm_nvtx3(INSTALL_EXPORT_SET test)

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_nvtx_dir/CMakeLists.txt"
     "
cmake_minimum_required(VERSION 3.30.4)
project(verify_nvtx LANGUAGES CXX)

set(CMAKE_PREFIX_PATH \"${CMAKE_BINARY_DIR}/install/\")
find_package(nvtx3 REQUIRED)

file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/stub.cpp\" \" \")
add_library(uses_nvtx SHARED stub.cpp)
target_link_libraries(uses_nvtx PRIVATE nvtx3::nvtx3-cpp)
")

add_custom_target(verify_build_config ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix ./install/
                  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/check_nvtx_dir/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_nvtx_dir"
                          -B="${CMAKE_BINARY_DIR}/build")
