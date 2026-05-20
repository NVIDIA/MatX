# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)

set(ENV{rapids_version} custom_env_version)
set(ENV{rapids_user} custom_env_user)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "nvbench": {
      "version": "$ENV{rapids_version}",
      "git_url": "$ENV{rapids_user}@gitlab.private.com",
      "git_tag": "my_tag"
    }
  }
}
  ]=])

rapids_cpm_init(OVERRIDE "${CMAKE_CURRENT_BINARY_DIR}/override.json")

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details_internal(nvbench version repository tag src_subdir shallow exclude)

if(NOT version STREQUAL "custom_env_version")
  message(FATAL_ERROR "custom version field was ignored. ${version} found instead of custom_env_version"
  )
endif()
if(NOT repository STREQUAL "custom_env_user@gitlab.private.com")
  message(FATAL_ERROR "custom git_url field was ignored. ${repository} found instead of custom_env_user@gitlab.private.com"
  )
endif()
if(NOT DEFINED CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be defined when an override exists")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be set to true when an override exists")
endif()
