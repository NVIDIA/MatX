# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init(GENERATE_PINNED_VERSIONS)
rapids_cpm_cccl(DOWNLOAD_ONLY ON)

add_custom_target(verify_generated_pins ALL
                  COMMAND ${CMAKE_COMMAND} "-S${CMAKE_SOURCE_DIR}/verify/"
                          "-B${CMAKE_BINARY_DIR}/verify_build"
                          "-Drapids-cmake-dir=${rapids-cmake-dir}"
                          "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}")
