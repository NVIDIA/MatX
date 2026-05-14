# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/get_override_json.cmake)

rapids_cpm_init()
rapids_cpm_package_override("${CMAKE_CURRENT_BINARY_DIR}/../rapids-cmake/pinned_versions.json")

# Read the pinned JSON for CCCL
get_override_json(CCCL pinned_json)

if(NOT pinned_json)
  message(FATAL_ERROR "CCCL should exist in pinned_versions.json")
endif()

# Verify url and url_hash are present
string(JSON url_value ERROR_VARIABLE no_url GET "${pinned_json}" url)
string(JSON url_hash_value ERROR_VARIABLE no_url_hash GET "${pinned_json}" url_hash)

if(no_url)
  message(FATAL_ERROR "Pinned CCCL entry should have 'url' field.\nPinned JSON: ${pinned_json}")
endif()
if(no_url_hash)
  message(FATAL_ERROR "Pinned CCCL entry should have 'url_hash' field.\nPinned JSON: ${pinned_json}"
  )
endif()

# Verify git_url and git_tag are NOT present (URL mode should not have git fields)
string(JSON git_url_value ERROR_VARIABLE no_git_url GET "${pinned_json}" git_url)
string(JSON git_tag_value ERROR_VARIABLE no_git_tag GET "${pinned_json}" git_tag)

if(NOT no_git_url)
  message(FATAL_ERROR "Pinned CCCL entry should NOT have 'git_url' field (URL mode package).\nPinned JSON: ${pinned_json}"
  )
endif()
if(NOT no_git_tag)
  message(FATAL_ERROR "Pinned CCCL entry should NOT have 'git_tag' field (URL mode package).\nPinned JSON: ${pinned_json}"
  )
endif()

# Verify the url contains the expected pattern (github tarball)
if(NOT url_value MATCHES "https://github.com/NVIDIA/cccl/archive/")
  message(FATAL_ERROR "Pinned CCCL url should be a GitHub tarball URL.\nGot: ${url_value}")
endif()

# Verify url_hash is a SHA256 hash
if(NOT url_hash_value MATCHES "^SHA256=")
  message(FATAL_ERROR "Pinned CCCL url_hash should be a SHA256 hash.\nGot: ${url_hash_value}")
endif()

message(STATUS "URL mode pinning verification passed for CCCL")
message(STATUS "  url: ${url_value}")
message(STATUS "  url_hash: ${url_hash_value}")
