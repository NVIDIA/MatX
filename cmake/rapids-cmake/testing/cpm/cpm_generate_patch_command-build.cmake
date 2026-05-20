# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
    "pkg_with_non_build_patch": {
      "version": "10.2",
      "git_url": "a_url",
      "git_tag": "a_tag",
      "patches": [
        {
          "file": "${current_json_dir}/example.diff",
          "issue": "explain",
          "fixed_in": "",
          "build": true
        },
        {
          "file": "${current_json_dir}/example2.diff",
          "issue": "explain",
          "fixed_in": ""
        }
      ]
    },
    "pkg_with_only_build_patches": {
      "version": "10.2",
      "git_url": "a_url",
      "git_tag": "a_tag",
      "patches": [
        {
          "file": "${current_json_dir}/example.diff",
          "issue": "explain",
          "fixed_in": "",
          "build": true
        },
        {
          "file": "${current_json_dir}/example2.diff",
          "issue": "explain",
          "fixed_in": "",
          "build": true
        }
      ]
    }
  }
}
  ]=])

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_BINARY_DIR}/override.json)

include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")

rapids_cpm_generate_patch_command(pkg_with_only_build_patches 10.2 patch_command build_patch_only)
if(NOT build_patch_only STREQUAL "BUILD_PATCH_ONLY")
  message(FATAL_ERROR "rapids_cpm_generate_patch_command generated wrong arguments for pkg_with_only_build_patches"
  )
endif()

rapids_cpm_generate_patch_command(pkg_with_non_build_patch 10.2 patch_command build_patch_only)
if(DEFINED build_patch_only)
  message(FATAL_ERROR "rapids_cpm_generate_patch_command generated wrong arguments for pkg_with_non_build_patch"
  )
endif()
