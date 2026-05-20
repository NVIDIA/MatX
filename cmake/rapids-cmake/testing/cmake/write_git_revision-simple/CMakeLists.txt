# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 2.4 LANGUAGES CXX)
rapids_cmake_write_git_revision_file(demo_version
                                     "${CMAKE_CURRENT_BINARY_DIR}/demo/demo_git_version.hpp")

project(NESTED VERSION 3.14.159 LANGUAGES NONE)
rapids_cmake_write_git_revision_file(nested_version
                                     "${CMAKE_CURRENT_BINARY_DIR}/nested/nested_git_version.h")

add_executable(write_git_version main.cpp)
target_link_libraries(write_git_version PRIVATE demo_version nested_version)
target_compile_features(write_git_version PRIVATE cxx_std_14)

install(TARGETS write_git_version EXPORT write_git)
rapids_export(BUILD DEMO EXPORT_SET write_git)

# write out the cmake file used to verify the copyright
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/verify_copyright.cmake
     [=[
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)
check_copyright_header("${WORKING_DIRECTORY}/demo/demo_git_version.hpp")
check_copyright_header("${WORKING_DIRECTORY}/nested/nested_git_version.h")

]=])

# Setup a custom build target that verifies that our generated git version header files have the
# correct copyright year
add_custom_target(verify_git_info ALL
                  COMMENT "verify copyright in generate git revision file for write_git_version"
                  COMMAND ${CMAKE_COMMAND} -DWORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
                          -Drapids-cmake-testing-dir=${rapids-cmake-testing-dir} -P
                          ${CMAKE_CURRENT_BINARY_DIR}/verify_copyright.cmake
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(verify_git_info demo_version_compute_git_info)
add_dependencies(verify_git_info nested_version_compute_git_info)
