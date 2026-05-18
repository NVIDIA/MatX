# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)

function(expect_fetch_content_details project expected)
  string(TOLOWER ${project} project)
  set(internal_fetch_content_var_name "_FetchContent_${project}_savedDetails")
  get_property(exists GLOBAL PROPERTY ${internal_fetch_content_var_name} SET)
  if(expected AND NOT exists)
    message(FATAL_ERROR "FetchContent expected ${project} doesn't match expected[${exists}!=${expected})"
    )
  elseif(NOT expected AND exists)
    message(FATAL_ERROR "FetchContent expected ${project} doesn't match expected[${exists}!=${expected})"
    )
  endif()
endfunction()

# Need to write out a default file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/default.json
     [=[
{
  "packages": {
    "nvbench": {
      "version": "0.0",
      "git_shallow": false,
      "git_url": "https://github.com/NVIDIA/nvbench.git",
      "git_tag": "555d628e9b250868c9da003e4407087ff1982e8e"
    },
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

# Load the default values for nvbench and GTest projects
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details_internal(nvbench nvbench_version nvbench_repository nvbench_tag
                                    nvbench_subdir nvbench_shallow nvbench_exclude)
rapids_cpm_package_details_internal(GTest GTest_version GTest_repository GTest_tag GTest_subdir
                                    GTest_shallow GTest_exclude)

expect_fetch_content_details(nvbench NO)
expect_fetch_content_details(rmm NO)
expect_fetch_content_details(GTest NO)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "nvbench": {
      "git_tag": "my_tag",
      "always_download": false
    },
    "rmm": {
      "git_tag": "my_tag"
    },
    "GTest": {
      "version": "2.99"
    }
  }
}
  ]=])

rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/default.json"
                OVERRIDE "${CMAKE_CURRENT_BINARY_DIR}/override.json")
expect_fetch_content_details(nvbench YES)
expect_fetch_content_details(rmm YES)
expect_fetch_content_details(GTest YES)

# Verify that the override works
rapids_cpm_package_details_internal(nvbench version repository tag src_subdir shallow exclude)
if(NOT version STREQUAL nvbench_version)
  message(FATAL_ERROR "default version field was removed.")
endif()
if(NOT repository STREQUAL nvbench_repository)
  message(FATAL_ERROR "default repository field was removed.")
endif()
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom git_tag field was ignored. ${tag} found instead of my_tag")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be false since the nvbench override explicitly sets it to 'false'"
  )
endif()
unset(CPM_DOWNLOAD_ALL)

# Verify that the override works
rapids_cpm_package_details_internal(rmm version repository tag src_subdir shallow exclude)
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom git_tag field was ignored. ${tag} found instead of my_tag")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be true since a custom git tag was used for rmm")
endif()
unset(CPM_DOWNLOAD_ALL)

rapids_cpm_package_details_internal(GTest version repository tag src_subdir shallow exclude)
if(NOT version STREQUAL "2.99")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "2.99")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead"
  )
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be false by default when an override exists that doesn't modify url or tag"
  )
endif()
