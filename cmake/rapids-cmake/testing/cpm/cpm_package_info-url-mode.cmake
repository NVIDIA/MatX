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

# Verify that rapids_cpm_package_info generates correct CPM arguments
include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")

# Test url package - should use URL and URL_HASH
rapids_cpm_package_info(test_url_pkg VERSION_VAR version CPM_VAR cpm_args)

# Verify URL mode args are present
foreach(expected_arg IN ITEMS URL URL_HASH)
  list(FIND cpm_args "${expected_arg}" arg_index)
  if(arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should contain '${expected_arg}' for url package.\nGot: ${cpm_args}"
    )
  endif()
endforeach()

# Verify git mode args are NOT present
foreach(unexpected_arg IN ITEMS GIT_REPOSITORY GIT_TAG GIT_SHALLOW)
  list(FIND cpm_args "${unexpected_arg}" arg_index)
  if(NOT arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should NOT contain '${unexpected_arg}' for url package.\nGot: ${cpm_args}"
    )
  endif()
endforeach()

# Verify the URL value is correct
list(FIND cpm_args "URL" url_index)
math(EXPR url_value_index "${url_index} + 1")
list(GET cpm_args ${url_value_index} url_value)
set(expected_url "https://github.com/NVIDIA/cccl/archive/abc123def456.tar.gz")
if(NOT url_value STREQUAL expected_url)
  message(FATAL_ERROR "URL value incorrect.\nExpected: ${expected_url}\nGot: ${url_value}")
endif()

# Verify the URL_HASH value is correct
list(FIND cpm_args "URL_HASH" url_hash_index)
math(EXPR url_hash_value_index "${url_hash_index} + 1")
list(GET cpm_args ${url_hash_value_index} url_hash_value)
set(expected_hash "SHA256=deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678")
if(NOT url_hash_value STREQUAL expected_hash)
  message(FATAL_ERROR "URL_HASH value incorrect.\nExpected: ${expected_hash}\nGot: ${url_hash_value}"
  )
endif()

# Test git package - should use GIT_REPOSITORY/GIT_TAG
rapids_cpm_package_info(test_git_pkg VERSION_VAR version CPM_VAR cpm_args)

# Verify git mode args are present
foreach(expected_arg IN ITEMS GIT_REPOSITORY GIT_TAG)
  list(FIND cpm_args "${expected_arg}" arg_index)
  if(arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should contain '${expected_arg}' for git package.\nGot: ${cpm_args}"
    )
  endif()
endforeach()

# Verify URL mode args are NOT present
foreach(unexpected_arg IN ITEMS URL URL_HASH)
  list(FIND cpm_args "${unexpected_arg}" arg_index)
  if(NOT arg_index EQUAL -1)
    message(FATAL_ERROR "CPM args should NOT contain '${unexpected_arg}' for git package.\nGot: ${cpm_args}"
    )
  endif()
endforeach()
