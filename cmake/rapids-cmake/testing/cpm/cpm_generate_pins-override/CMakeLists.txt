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

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages" : {
    "rmm" : {
      "version" : "22.02",
      "git_url" : "https://github.com/rapidsai/rmm.git",
      "git_tag" : "v22.02.00"
    }
  }
}
  ]=])
include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

add_custom_target(verify_generated_pins ALL
                  COMMAND ${CMAKE_COMMAND} "-S${CMAKE_SOURCE_DIR}/verify/"
                          "-B${CMAKE_BINARY_DIR}/verify_build"
                          "-Drapids-cmake-dir=${rapids-cmake-dir}"
                          "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}")
