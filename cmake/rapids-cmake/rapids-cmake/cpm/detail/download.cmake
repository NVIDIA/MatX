#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_download
-------------------

.. versionadded:: v21.10.00

Does the downloading of the `CPM` module

.. code-block:: cmake

  rapids_cpm_download()

The CPM module will be downloaded based on the state of :cmake:variable:`CPM_SOURCE_CACHE` and
:cmake:variable:`ENV{CPM_SOURCE_CACHE}`.

.. note::
  Use `rapids_cpm_init` instead of this function, as this is an implementation detail
  required for proper cpm project exporting in build directories

  This function can't call other rapids-cmake functions, due to the
  restrictions of `write_dependencies.cmake`

#]=======================================================================]
function(rapids_cpm_download)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.download")

  # When changing version verify no new variables needs to be propagated
  set(CPM_DOWNLOAD_VERSION 0.38.5)
  set(CPM_DOWNLOAD_MD5_HASH c98d14a13dfd1952e115979c095f6794)

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

  if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(VERBOSE "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
         https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
         ${CPM_DOWNLOAD_LOCATION} LOG download_log)

    file(MD5 ${CPM_DOWNLOAD_LOCATION} cpm_hash)
    if(NOT cpm_hash STREQUAL CPM_DOWNLOAD_MD5_HASH)
      message(FATAL_ERROR "CPM.cmake hash mismatch [got=${cpm_hash} expected=${CPM_DOWNLOAD_MD5_HASH}] to download details below\n ${download_log}"
      )
    endif()
  endif()

  include(${CPM_DOWNLOAD_LOCATION})

  # Propagate up any modified local variables that CPM has changed.
  #
  # Push up the modified CMAKE_MODULE_PATh to allow `find_package` calls to find packages that CPM
  # already added.
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

endfunction()
