# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file with url and url_hash
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "test_url_pkg": {
      "version": "1.0.0",
      "url": "https://github.com/NVIDIA/cccl/archive/abc123def456.tar.gz",
      "url_hash": "SHA256=deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
    },
    "test_git_pkg": {
      "version": "2.0.0",
      "git_url": "https://github.com/NVIDIA/other.git",
      "git_tag": "xyz789"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that url/url_hash mode works correctly
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

rapids_cpm_package_details_internal(test_url_pkg version url tag src_subdir shallow exclude)

# Verify URL is passed through correctly
set(expected_url "https://github.com/NVIDIA/cccl/archive/abc123def456.tar.gz")
if(NOT url STREQUAL expected_url)
  message(FATAL_ERROR "url should be passed through.\nExpected: ${expected_url}\nGot: ${url}")
endif()

# Verify tag is empty (signals URL-based fetching)
if(tag)
  message(FATAL_ERROR "tag should be empty for url mode. Got: '${tag}'")
endif()

# Verify url_hash is set in parent scope
if(NOT _rapids_url_hash STREQUAL
   "SHA256=deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678")
  message(FATAL_ERROR "_rapids_url_hash should be set. Got: '${_rapids_url_hash}'")
endif()

# Verify version is still correct
if(NOT version STREQUAL "1.0.0")
  message(FATAL_ERROR "version should be 1.0.0. Got: ${version}")
endif()

# Verify that git_url/git_tag packages still work
rapids_cpm_package_details_internal(test_git_pkg version url tag src_subdir shallow exclude)

if(NOT url STREQUAL "https://github.com/NVIDIA/other.git")
  message(FATAL_ERROR "git_url should be preserved for git packages. Got: ${url}")
endif()

if(NOT tag STREQUAL "xyz789")
  message(FATAL_ERROR "git_tag should be preserved for git packages. Got: ${tag}")
endif()
