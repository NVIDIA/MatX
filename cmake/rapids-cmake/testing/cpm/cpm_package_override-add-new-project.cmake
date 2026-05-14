# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "custom_package_never_in_rapids" : {
      "version" : "3.1.0",
      "git_url" : "https://github.com/NVIDIA/NVTX",
      "git_tag" : "96aeb0d8702981972fac0f6e485fea7acbfd5446",
      "git_shallow" : false,
      "exclude_from_all" : true
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")

rapids_cpm_package_info(custom_package_never_in_rapids VERSION_VAR version)
if(NOT version STREQUAL "3.1.0")
  message(FATAL_ERROR "expected version field wasn't found. ${version} was found instead")
endif()

# Make sure we can clone without getting an error when parsing the EXCLUDE_FROM_ALL tag
rapids_cpm_find(custom_package_never_in_rapids 3.1)
