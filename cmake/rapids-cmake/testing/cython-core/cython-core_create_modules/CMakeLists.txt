# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.20)

include(${rapids-cmake-dir}/cython-core/create_modules.cmake)
include(${rapids-cmake-dir}/cython-core/init.cmake)

project(rapids_cython-create_modules LANGUAGES C CXX)

rapids_cython_init()

# Test that an empty invocation works.
rapids_cython_create_modules()

# Test that a basic invocation with a single file works.
rapids_cython_create_modules(SOURCE_FILES test.pyx)

if(NOT TARGET test)
  message(FATAL_ERROR "rapids_cython_create_modules didn't create the target `test`")
endif()

# Test that the dependency file was created properly.
cmake_path(SET depfile NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/test.c.dep")
add_custom_target(check_depfile_exists ALL DEPENDS test)
add_custom_command(TARGET check_depfile_exists
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -DDEPFILE="${depfile}" -P
                           ${CMAKE_CURRENT_SOURCE_DIR}/check_depfile_exists.cmake
                   COMMENT "Checking for existence of dependency file")
