# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-cpm-find-add-pkg-source LANGUAGES CXX)

set(CPM_ZLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/mock_zlib_source_dir")

rapids_cpm_init()
rapids_cpm_find(ZLIB 1.0)

if(NOT TARGET MOCK_ZLIB)
  message(FATAL_ERROR "rapids_cpm_find failed to add ZLIB source dir ${CPM_ZLIB_SOURCE}")
endif()

if(NOT DEFINED ZLIB_SOURCE_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_SOURCE_DIR variable")
endif()
if(NOT DEFINED ZLIB_BINARY_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_BINARY_DIR variable")
endif()
if(NOT DEFINED ZLIB_ADDED)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_ADDED variable")
endif()

if(NOT ZLIB_ADDED)
  message(FATAL_ERROR "rapids_cpm_find failed to find local mock zlib")
endif()

set(old_source_dir ${ZLIB_SOURCE_DIR})
set(old_binary_dir ${ZLIB_BINARY_DIR})
unset(ZLIB_SOURCE_DIR)
unset(ZLIB_BINARY_DIR)
unset(ZLIB_ADDED)

rapids_cpm_find(ZLIB 1.0)

if(NOT DEFINED ZLIB_SOURCE_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_SOURCE_DIR variable")
endif()
if(NOT DEFINED ZLIB_BINARY_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_BINARY_DIR variable")
endif()
if(NOT DEFINED ZLIB_ADDED)
  message(FATAL_ERROR "rapids_cpm_find failed to construct ZLIB_ADDED variable")
endif()

if(NOT old_source_dir STREQUAL ZLIB_SOURCE_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to restore correct ZLIB_SOURCE_DIR path")
endif()
if(NOT old_binary_dir STREQUAL ZLIB_BINARY_DIR)
  message(FATAL_ERROR "rapids_cpm_find failed to restore correct ZLIB_BINARY_DIR path")
endif()
if(ZLIB_ADDED)
  message(FATAL_ERROR "rapids_cpm_find failed to restore correct ZLIB_ADDED state ( second call should be ZLIB_ADDED=false )"
  )
endif()
