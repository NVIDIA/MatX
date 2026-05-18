# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.23)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

project(fake LANGUAGES CXX VERSION 3.1.4)

rapids_cpm_init()

include(GNUInstallDirs)

function(call_cccl_with_custom_install_loc)
  set(CMAKE_INSTALL_LIBDIR "custom/lib/aarch64")
  set(CMAKE_INSTALL_INCLUDEDIR "custom/include/aarch64")
  rapids_cpm_cccl(INSTALL_EXPORT_SET export_set)
endfunction()

# We will install with the custom install location set in this file
call_cccl_with_custom_install_loc()

# Call for `cccl` again but ensure we don't re-invoke the install rules overwriting the custom
# install location in files like thrust-header-search.cmake
rapids_cpm_cccl(INSTALL_EXPORT_SET export_set)

# Install our project so we can verify `cccl`/`thrust` has preserved the correct install location
add_custom_target(install_project ALL COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
                                              --prefix check_thrust/install/)

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_thrust/CMakeLists.txt"
     [=[
cmake_minimum_required(VERSION 3.20)
project(verify_install_targets LANGUAGES CXX)

set(computed_dir "${CMAKE_CURRENT_SOURCE_DIR}/install/custom/lib/aarch64/rapids/cmake/thrust/")
set(header_search_file "${computed_dir}/thrust-header-search.cmake")

if(NOT EXISTS ${header_search_file})
  message(FATAL_ERROR "Failed to compute thrust-header-search.cmake location")
endif()

include("${header_search_file}")
# The `from_install_prefix` should be "custom/lib/aarch64/rapids/cmake/thrust" which
# will be transformed into `../../../../../../`
set(expected "../../../../../../")
if(NOT from_install_prefix STREQUAL expected)
  message(FATAL_ERROR "thrust-header-search.cmake contains the wrong 'from_install_prefix'. Expected '${expected}' but found '${from_install_prefix}'")
endif()
]=])

add_custom_target(verify_thrust_header_search ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/check_thrust/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_thrust"
                          -B="${CMAKE_BINARY_DIR}/install/build")
add_dependencies(verify_thrust_header_search install_project)
