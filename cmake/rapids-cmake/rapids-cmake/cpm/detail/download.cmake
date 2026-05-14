# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_download
-------------------

.. versionadded:: v21.10.00

Does the downloading of the `CPM` module

.. code-block:: cmake

  rapids_cpm_download()

The CPM module will be downloaded based on the following.

.. versionadded:: v24.10.00

If :cmake:variable:`CPM_DOWNLOAD_LOCATION` is defined that location will be used
as the download location. If a file already exists at that location no download will occur

If the :cmake:variable:`CPM_SOURCE_CACHE` or :cmake:variable:`ENV{CPM_SOURCE_CACHE}` are
defined those will be used to compute a location for the file.

If none of the above variables are defined, rapids-cmake will download the file
to `cmake` directory under :cmake:variable:`CMAKE_BINARY_DIR`.

.. note::
  Use `rapids_cpm_init` instead of this function, as this is an implementation detail
  required for proper cpm project exporting in build directories

  This function can't call other rapids-cmake functions, due to the
  restrictions of `write_dependencies.cmake`

#]=======================================================================]
function(rapids_cpm_download)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.download")

  # When changing version verify no new variables needs to be propagated
  set(CPM_DOWNLOAD_VERSION 0.42.0)
  set(CPM_DOWNLOAD_SHA256_HASH 2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a)

  if(NOT DEFINED CPM_DOWNLOAD_LOCATION)
    if(CPM_SOURCE_CACHE)
      # Expand relative path. This is important if the provided path contains a tilde (~)
      cmake_path(ABSOLUTE_PATH CPM_SOURCE_CACHE)

      # default to the same location that cpm computes
      set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      if(EXISTS "${CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        # Also support the rapids-cmake download location ( cmake/ vs cpm/ )
        set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      endif()

    elseif(DEFINED ENV{CPM_SOURCE_CACHE})

      # default to the same location that cpm computes
      set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      if(EXISTS "$ENV{CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        # Also support the rapids-cmake download location ( cmake/ vs cpm/ )
        set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
      endif()

    else()
      set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    endif()
  endif()

  if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(VERBOSE "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    if(NOT COMMAND rapids_cmake_download_with_retry)
      include("${rapids-cmake-dir}/cmake/download_with_retry.cmake")
    endif()
    rapids_cmake_download_with_retry(
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
      ${CPM_DOWNLOAD_LOCATION} ${CPM_DOWNLOAD_SHA256_HASH})
  endif()

  include(${CPM_DOWNLOAD_LOCATION})

  # Propagate up any modified local variables that CPM has changed.
  #
  # Push up the modified CMAKE_MODULE_PATh to allow `find_package` calls to find packages that CPM
  # already added.
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

endfunction()
