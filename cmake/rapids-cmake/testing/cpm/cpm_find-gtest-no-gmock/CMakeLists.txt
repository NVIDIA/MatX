# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.23)
project(rapids-cpm-find-gtest-no-gmock LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mock_installed_gtest")
# Downloading is turned on by default for all packages in the CPM cache when testing, but in this
# case we specifically want it off.
set(CPM_DOWNLOAD_GTest OFF)

rapids_cpm_init()
rapids_cpm_gtest()

if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR "rapids_cpm_find failed to add mocked gtest version")
endif()

if(TARGET GTest::gmock)
  message(FATAL_ERROR "rapids_cpm_find failed to add mocked gtest version")
endif()
