# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)
project(rapids-cmake-testing LANGUAGES NONE)

enable_testing()
include(utils/cmake_config_test.cmake)
include(utils/cmake_ctest_test.cmake)
include(utils/cmake_build_test.cmake)

add_subdirectory(cmake)
add_subdirectory(cuda)
add_subdirectory(export)
add_subdirectory(find)

include("${CMAKE_CURRENT_SOURCE_DIR}/utils/determine_gpu_count.cmake")
determine_gpu_count()
add_subdirectory(test)

option(RAPIDS_CMAKE_ENABLE_DOWNLOAD_TESTS "Enable tests that require downloads" ON)
if(RAPIDS_CMAKE_ENABLE_DOWNLOAD_TESTS)
  # Setup and fill the cpm cache, and specify CPM_SOURCE_CACHE so that all subsequent tests in this
  # directory use the shared CPM_SOURCE_CACHE
  include("${CMAKE_CURRENT_SOURCE_DIR}/utils/setup_cpm_cache.cmake")
  setup_cpm_cache()

  add_subdirectory(cpm)
  add_subdirectory(cython-core)
  add_subdirectory(other)
endif()
