# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override1.json
     [=[
{
  "packages": {
    "nvbench": {
      "git_tag": "my_tag"
    },
    "gtest": {
      "version": "2.99",
      "git_tag": "v${version}"
    }
  }
}
  ]=])

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override2.json
     [=[
{
  "packages": {
    "rmm": {
      "git_tag": "new_rmm_tag"
    },
    "GTest": {
      "version": "3.99",
      "git_tag": "v${version}"
    }
  }
}
  ]=])

set(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/override1.json")
rapids_cpm_init()
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override2.json)

# Verify that the override works
rapids_cpm_package_details_internal(nvbench version repository tag src_subdir shallow exclude)
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom git_tag field was ignored. ${tag} found instead of my_url")
endif()

rapids_cpm_package_details_internal(GTest version repository tag src_subdir shallow exclude)
if(NOT version STREQUAL "2.99")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "2.99")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead"
  )
endif()

rapids_cpm_package_details_internal(rmm version repository tag src_subdir shallow exclude)
if(NOT tag MATCHES "new_rmm_tag")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead"
  )
endif()
