# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/cccl.cmake)
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_INCLUDEDIR "include")

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "CCCL": {
      "version": "3.0.0",
      "git_shallow": false,
      "git_url": "https://github.com/NVIDIA/cccl.git",
      "git_tag": "dc1db8aefcfe52c66c26c6038fef58da37a46ae9"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

rapids_cpm_cccl(INSTALL_EXPORT_SET export_set)

# Install our project so we can verify `cccl`/`thrust` has preserved the correct install location
add_custom_target(install_project ALL COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
                                              --prefix check_cccl/install/)

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_cccl/CMakeLists.txt"
     [=[
cmake_minimum_required(VERSION 3.20)
project(verify_install_targets LANGUAGES CXX)

# Verify include dirs
set(include_dirs_to_verify include/rapids/cub
                           include/rapids/cuda
                           include/rapids/nv
                           include/rapids/thrust)

# Verify lib/cmake dirs
set(cmake_dirs_to_verify lib/rapids/cmake/cccl
                         lib/rapids/cmake/cub
                         lib/rapids/cmake/libcudacxx
                         lib/rapids/cmake/thrust)

foreach(to_verify IN LISTS include_dirs_to_verify cmake_dirs_to_verify)
  set(path "${CMAKE_CURRENT_SOURCE_DIR}/install/${to_verify}")
  if(NOT EXISTS ${path})
    message(FATAL_ERROR "Failed to find `${path}` location")
  endif()
endforeach()
]=])

add_custom_target(verify_thrust_header_search ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/check_cccl/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_cccl"
                          -B="${CMAKE_BINARY_DIR}/install/build")
add_dependencies(verify_thrust_header_search install_project)
