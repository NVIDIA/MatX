# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cmake_download_with_retry
--------------------------------

.. versionadded:: v25.06.00

Downloads a file from a URL with retry logic for handling network issues.

  .. code-block:: cmake

    rapids_cmake_download_with_retry(url output_file sha256 [MAX_RETRIES <max_retries>] [RETRY_DELAY <retry_delay>])

This function will attempt to download the file multiple times if network issues occur.
It verifies the download by checking the SHA256 checksum of the downloaded file. If all
retries fail, it will raise a fatal error.

``url``
  The URL to download from.

``output_file``
  The path where the downloaded file should be saved.

``sha256``
  The expected SHA256 checksum of the file.

``MAX_RETRIES``
  Maximum number of retry attempts. Defaults to 10.

``RETRY_DELAY``
  Delay between retries in seconds. Defaults to 5.

#]=======================================================================]
function(rapids_cmake_download_with_retry url output_file sha256)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.download_with_retry")

  set(options)
  set(one_value MAX_RETRIES RETRY_DELAY)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Set default values for optional arguments
  if(NOT DEFINED _RAPIDS_MAX_RETRIES)
    set(_RAPIDS_MAX_RETRIES 10)
  endif()
  if(NOT DEFINED _RAPIDS_RETRY_DELAY)
    set(_RAPIDS_RETRY_DELAY 5)
  endif()

  # Set up retry parameters
  set(current_retry 0)
  set(download_success FALSE)

  while(NOT download_success AND current_retry LESS ${_RAPIDS_MAX_RETRIES})
    if(current_retry GREATER 0)
      message(STATUS "Retrying download (attempt ${current_retry} of ${_RAPIDS_MAX_RETRIES}) after ${_RAPIDS_RETRY_DELAY} seconds..."
      )
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${_RAPIDS_RETRY_DELAY})
    endif()

    # Remove any existing file to ensure clean download
    if(EXISTS "${output_file}")
      file(REMOVE "${output_file}")
    endif()

    file(DOWNLOAD "${url}" "${output_file}" LOG download_log)

    # Check if file exists and validate SHA256
    if(EXISTS "${output_file}")
      file(SHA256 "${output_file}" downloaded_sha256)
      if(downloaded_sha256 STREQUAL "${sha256}")
        set(download_success TRUE)
      else()
        message(WARNING "Downloaded file SHA256 checksum mismatch. Expected: ${sha256}, Got: ${downloaded_sha256}"
        )
        file(REMOVE "${output_file}")
      endif()
    endif()

    if(NOT download_success)
      math(EXPR current_retry "${current_retry} + 1")
      if(current_retry LESS ${_RAPIDS_MAX_RETRIES})
        message(WARNING "Failed to download file. Will retry. Download log:\n${download_log}")
      else()
        message(FATAL_ERROR "Failed to download file after ${_RAPIDS_MAX_RETRIES} attempts. Download log:\n${download_log}"
        )
      endif()
    endif()
  endwhile()
endfunction()
