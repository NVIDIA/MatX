# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include("${rapids-cmake-testing-dir}/cpm/make_fake_project_build_dir_with_config.cmake")

make_fake_project_build_dir_with_config(RapidsTestFind 2021.01.02 RapidsTestFindConfig.cmake
                                        RapidsTestFindConfigVersion.cmake)

make_fake_project_build_dir_with_config(RapidsTestFind2 0.2 rapidstestfind2-config.cmake
                                        rapidstestfind2-config-version.cmake)

rapids_cpm_init()

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/RapidsTestFind-build/"
                      "${CMAKE_CURRENT_BINARY_DIR}/RapidsTestFind2-build/")

rapids_cpm_find(RapidsTestFind 2021.01.02)
rapids_cpm_find(RapidsTestFind2 0.2)

if(NOT TARGET RapidsTest::RapidsTest)
  message(FATAL_ERROR "RapidsTest targets should be generated")
endif()
if(NOT TARGET RapidsTest2::RapidsTest)
  message(FATAL_ERROR "RapidsTest2 targets should be generated")
endif()
