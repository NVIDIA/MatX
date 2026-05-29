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

# Test that we can use a prefix to generate a different library target.
rapids_cython_create_modules(SOURCE_FILES test.pyx MODULE_PREFIX "test_")

if(NOT TARGET test_test)
  message(FATAL_ERROR "rapids_cython_create_modules didn't create the prefixed library target `test_test`"
  )
endif()
