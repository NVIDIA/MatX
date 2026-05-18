# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

# Write a default file with git mode
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/default.json
     [=[
{
  "packages": {
    "test_pkg": {
      "version": "1.0.0",
      "git_url": "https://github.com/NVIDIA/cccl.git",
      "git_tag": "v1.0.0"
    }
  }
}
  ]=])

rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/default.json")

# Write an override file that switches from git mode to url mode
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "test_pkg": {
      "url": "https://github.com/NVIDIA/cccl/archive/abc123.tar.gz",
      "url_hash": "SHA256=deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override switched to url mode
include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
rapids_cpm_package_info(test_pkg VERSION_VAR version CPM_VAR cpm_args)

# Verify URL mode args are present
foreach(expected_arg IN ITEMS URL URL_HASH)
  list(FIND cpm_args "${expected_arg}" arg_index)
  if(arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should contain '${expected_arg}' after override to url mode.\nGot: ${cpm_args}"
    )
  endif()
endforeach()

# Verify git mode args are NOT present
foreach(unexpected_arg IN ITEMS GIT_REPOSITORY GIT_TAG GIT_SHALLOW)
  list(FIND cpm_args "${unexpected_arg}" arg_index)
  if(NOT arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should NOT contain '${unexpected_arg}' after override to url mode.\nGot: ${cpm_args}"
    )
  endif()
endforeach()
