# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-cpm_find-patch-command-project LANGUAGES CXX)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "gtest": {
      "patches": [
        {
          "file": "${current_json_dir}/patches/0001-move-git-sha1.patch",
          "issue": "Move git sha1",
          "fixed_in": ""
        }
      ]
    }
  }
}
  ]=])

include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
rapids_cpm_package_info(GTest VERSION_VAR version)

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_BINARY_DIR}/override.json)

include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
rapids_cpm_generate_patch_command(GTest ${version} patch_command build_patch_only)
message(STATUS "patch_command: ${patch_command}")
if(NOT patch_command)
  message(FATAL_ERROR "rapids_cpm_package_override failed to load patch step for `GTest` from package `gtest`"
  )
endif()

# Need to load ${build_dir}/rapids-cmake/patches/GTest/patch.cmake and verify that the `files`
# variable has properly resolved `current_json_dir`
set(patch_script "${CMAKE_BINARY_DIR}/rapids-cmake/patches/GTest/patch.cmake")
include("${patch_script}")
if(files STREQUAL "/patches/0001-move-git-sha1.patch")
  message(FATAL_ERROR "rapids_cpm_package_override failed to properly expand 'current_json_dir'")
endif()
