# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

# Need to write out a default file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/default.json
     [=[
{
  "packages": {
    "rmm": {
      "version": "${rapids-cmake-version}",
      "git_url": "https://github.com/rapidsai/rmm.git",
      "git_tag": "${rapids-cmake-checkout-tag}"
    },
    "GTest": {
      "version": "1.16.0",
      "git_url": "https://github.com/google/googletest.git",
      "git_tag": "v${version}"
    }
  }
}
  ]=])

rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/default.json")

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "rMm": {
      "git_url": "new_rmm_url",
      "git_shallow": "OFF",
      "exclude_from_all": "ON"
    },
    "gtest": {
      "version": "3.00.A1"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

rapids_cpm_package_details_internal(GTest version repository tag src_subdir shallow exclude)
if(NOT version STREQUAL "3.00.A1")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "3.00.A1")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead"
  )
endif()
if(NOT exclude MATCHES "OFF")
  message(FATAL_ERROR "default value of exclude not found. ${exclude} was found instead")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be false by default when an override exists that doesn't modify url or tag"
  )
endif()

rapids_cpm_package_details_internal(rmm version repository tag src_subdir shallow exclude)
if(NOT repository MATCHES "new_rmm_url")
  message(FATAL_ERROR "custom url field was not used. ${repository} was found instead")
endif()
if(NOT shallow MATCHES "OFF")
  message(FATAL_ERROR "override should not change git_shallow value. ${shallow} was found instead")
endif()
if(NOT exclude MATCHES "ON")
  message(FATAL_ERROR "override should have changed exclude value. ${exclude} was found instead")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be set to true when an override exists with a custom repository"
  )
endif()
