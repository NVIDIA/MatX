# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
      "git_url": "new_rmm_url",
      "git_shallow": "OFF",
      "exclude_from_all": "ON"
    },
    "not_in_base": {
      "version": "1.0",
      "git_url": "new_rmm_url",
      "git_tag": "main",
      "git_shallow": "OFF",
      "exclude_from_all": "ON"
    }
  }
}
  ]=])

set(CPM_rmm_SOURCE "${CMAKE_CURRENT_BINARY_DIR}")
set(CPM_not_in_base_SOURCE "${CMAKE_CURRENT_BINARY_DIR}")
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override doesn't exist due to `CPM_rmm_SOURCE`
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

rapids_cpm_package_details_internal(rmm version repository tag src_subdir shallow exclude)
if(NOT repository MATCHES "new_rmm_url")
  message(FATAL_ERROR "custom url field should not be set, due to CPM_rmm_SOURCE")
endif()
if(NOT shallow MATCHES "OFF")
  message(FATAL_ERROR "shallow field should not be set, due to CPM_rmm_SOURCE")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should not be set, due to CPM_rmm_SOURCE")
endif()

unset(version)
unset(repository)
unset(tag)
rapids_cpm_package_details_internal(not_in_base version repository tag src_subdir shallow exclude)
if(NOT (version AND repository AND tag))
  message(FATAL_ERROR "rapids_cpm_package_details should still have details for package that doesn't exist"
  )
endif()

get_property(override_ignored GLOBAL PROPERTY rapids_cpm_rmm_override_ignored)
if(NOT override_ignored)
  message(FATAL_ERROR "rapids_cpm_package override for `not_in_base` isn't being ignored")
endif()

get_property(override_ignored GLOBAL PROPERTY rapids_cpm_not_in_base_override_ignored)
if(NOT override_ignored)
  message(FATAL_ERROR "rapids_cpm_package override for `not_in_base` isn't being ignored")
endif()
