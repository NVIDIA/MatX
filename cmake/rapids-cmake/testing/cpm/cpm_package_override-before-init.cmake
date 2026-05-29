# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/simple_override.json
     [=[
{
  "packages": {
    "nvbench": {
      "version": "custom_version",
      "git_url": "my_url2",
      "git_tag": "my_tag"
    }
  }
}
  ]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/simple_override.json)

rapids_cpm_init()

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details_internal(nvbench version repository tag src_subdir shallow exclude)

if(NOT version STREQUAL "custom_version")
  message(FATAL_ERROR "custom version field was ignored. ${version} found instead of custom_version"
  )
endif()
if(NOT repository STREQUAL "my_url2")
  message(FATAL_ERROR "custom git_url field was ignored. ${repository} found instead of my_url2")
endif()
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom git_tag field was ignored. ${tag} found instead of my_tag")
endif()
