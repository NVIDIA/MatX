# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.26.4)
project(rapids-test-project LANGUAGES CXX)

set(CPM_cucxx_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/mock_cucxx_source_dir")

rapids_cpm_init()
rapids_cpm_find(cucxx 1.0)

# Should find CPM generated `Findcucxx.cmake`
find_package(cucxx REQUIRED)

set(expected_find_path "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/cucxx-config.cmake")
if(NOT EXISTS "${expected_find_path}")
  message(FATAL_ERROR "cucxx-config.cmake was not generated")
endif()

if(NOT TARGET MOCK_CUCXX)
  message(FATAL_ERROR "cucxx targets should be generated")
endif()
