# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)

# Need to write out a custom default file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/defaults.json
     [=[
{
  "packages": {
    "nvbench": {
      "version": "custom_version",
      "git_url": "my_url",
      "git_tag": "my_tag"
    }
  }
}
  ]=])

rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/defaults.json")

# Verify that the custom defaults works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details_internal(nvbench version repository tag src_subdir shallow exclude)

if(NOT version STREQUAL "custom_version")
  message(FATAL_ERROR "custom default version field was ignored. ${version} found instead of custom_version"
  )
endif()
if(NOT repository STREQUAL "my_url")
  message(FATAL_ERROR "custom default git_url field was ignored. ${repository} found instead of my_url"
  )
endif()
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom default git_tag field was ignored. ${tag} found instead of my_tag")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL shouldn't be set to true when using a custom default")
endif()
