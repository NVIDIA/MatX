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
