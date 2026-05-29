# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/download_with_retry.cmake)

# Create a test directory
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/download_test")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/download_test")

# Test URL - using a static test file with known contents
set(test_url
    "https://raw.githubusercontent.com/rapidsai/rapids-cmake/c0d8c09c5590ecf38a9f9897c93e686e3da1858b/testing/cmake/test_files/test1.txt"
)
set(output_file "${CMAKE_CURRENT_BINARY_DIR}/download_test/test_file.txt")
# Expected SHA256 of the test file (content: "This is a test file for rapids-cmake download tests.")
set(expected_sha256 "7a90fe28cdb40b030ad3323af3fd9292d849db0be1d79a7000f9ff150c00161f")

# Test 1: Basic download
rapids_cmake_download_with_retry("${test_url}" "${output_file}" "${expected_sha256}")
if(NOT EXISTS "${output_file}")
  message(FATAL_ERROR "Download failed - file does not exist")
endif()

# Verify file has content
file(SIZE "${output_file}" file_size)
if(file_size EQUAL 0)
  message(FATAL_ERROR "Downloaded file is empty")
endif()

# Test 2: Download with custom retry parameters
set(output_file2 "${CMAKE_CURRENT_BINARY_DIR}/download_test/test_file2.txt")
rapids_cmake_download_with_retry("${test_url}" "${output_file2}" "${expected_sha256}" MAX_RETRIES 2
                                 RETRY_DELAY 1)
if(NOT EXISTS "${output_file2}")
  message(FATAL_ERROR "Download with custom parameters failed - file does not exist")
endif()

# Verify second file has content
file(SIZE "${output_file2}" file_size)
if(file_size EQUAL 0)
  message(FATAL_ERROR "Second downloaded file is empty")
endif()
