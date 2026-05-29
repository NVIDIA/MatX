# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake)
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "pkg_with_patch": {
      "version": "10.2",
      "git_url" : "a_url",
      "git_tag": "a_tag",
      "git_shallow": "OFF",
      "exclude_from_all": "ON",
      "patches": [
        {
          "file": "e/example.diff",
          "issue": "explain",
          "fixed_in": ""
        }
      ]
    }
  }
}
  ]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

rapids_cpm_generate_patch_command(pkg_with_patch 10.2 patch_command build_patch_only)
if(NOT patch_command)
  message(FATAL_ERROR "rapids_cpm_package_override specified a patch step for `pkg_with_patch`")
endif()

check_copyright_header("${CMAKE_BINARY_DIR}/rapids-cmake/patches/pkg_with_patch/patch.cmake")
