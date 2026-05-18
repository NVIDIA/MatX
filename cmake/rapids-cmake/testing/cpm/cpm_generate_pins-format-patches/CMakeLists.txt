# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init(GENERATE_PINNED_VERSIONS)

include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override(${CMAKE_CURRENT_SOURCE_DIR}/override.json)

include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(DOWNLOAD_ONLY ON)

# Verify that the two files that we inserted into the RMM source tree exist which proves the patches
# in the override are properly applied
if(NOT EXISTS "${rmm_SOURCE_DIR}/git_file_1.txt")
  message(FATAL_ERROR "failed to apply rmm first patch")
endif()

if(NOT EXISTS "${rmm_SOURCE_DIR}/git_file_2.txt")
  message(FATAL_ERROR "failed to apply rmm second patch")
endif()

# Verify that the one file that we inserted into the cuco source tree exists.
if(NOT EXISTS "${cuco_SOURCE_DIR}/git_file_1.txt")
  message(FATAL_ERROR "failed to apply cucco first patch")
endif()

add_custom_target(verify_generated_pins ALL
                  COMMAND ${CMAKE_COMMAND} "-S${CMAKE_SOURCE_DIR}/verify/"
                          "-B${CMAKE_BINARY_DIR}/verify_build"
                          "-Drapids-cmake-dir=${rapids-cmake-dir}"
                          "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}")
