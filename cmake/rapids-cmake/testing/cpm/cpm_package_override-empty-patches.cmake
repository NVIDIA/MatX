# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "rmm": {
      "patches": [
        {
        }
      ]
    },
    "CCCL": {
      "patches": [ ]
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")

rapids_cpm_package_details_internal(rmm version repository tag src_subdir shallow exclude)
rapids_cpm_generate_patch_command(rmm ${version} patch_command build_patch_only)
if(patch_command)
  message(FATAL_ERROR "no patch command expected for rmm")
endif()

rapids_cpm_package_details_internal(CCCL version repository tag src_subdir shallow exclude)
rapids_cpm_generate_patch_command(CCCL ${version} patch_command build_patch_only)
if(patch_command)
  message(FATAL_ERROR "no patch command expected for cccl")
endif()
