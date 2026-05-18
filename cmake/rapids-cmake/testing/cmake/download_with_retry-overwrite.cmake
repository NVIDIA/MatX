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

# Test URLs - using static test files with known contents
set(test_url1
    "https://raw.githubusercontent.com/rapidsai/rapids-cmake/c0d8c09c5590ecf38a9f9897c93e686e3da1858b/testing/cmake/test_files/test1.txt"
)
set(test_url2
    "https://raw.githubusercontent.com/rapidsai/rapids-cmake/c0d8c09c5590ecf38a9f9897c93e686e3da1858b/testing/cmake/test_files/test2.txt"
)
set(output_file "${CMAKE_CURRENT_BINARY_DIR}/download_test/overwrite_test.txt")

# Expected SHA256 values for the test files:

# test1.txt content: "This is a test file for rapids-cmake download tests."

# test2.txt content: "This is a different test file for rapids-cmake download tests."

set(expected_sha256_1 "7a90fe28cdb40b030ad3323af3fd9292d849db0be1d79a7000f9ff150c00161f")
set(expected_sha256_2 "d3d4925f1a6596154ddd957ea16d5350f229a369ef24574563ceecca2ac8c66c")

# Test 1: Create initial file with first URL
rapids_cmake_download_with_retry("${test_url1}" "${output_file}" "${expected_sha256_1}")
if(NOT EXISTS "${output_file}")
  message(FATAL_ERROR "Initial download failed - file does not exist")
endif()

# Store initial file size
file(SIZE "${output_file}" initial_size)

# Test 2: Download different file to same location
rapids_cmake_download_with_retry("${test_url2}" "${output_file}" "${expected_sha256_2}")
if(NOT EXISTS "${output_file}")
  message(FATAL_ERROR "Overwrite download failed - file does not exist")
endif()

# Verify file was overwritten with new content
file(SIZE "${output_file}" new_size)
if(new_size EQUAL initial_size)
  message(FATAL_ERROR "File was not properly overwritten - sizes are identical")
endif()
