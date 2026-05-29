# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
download_proprietary_binary
-------------------

.. versionadded:: v23.04.00

Download the associated proprietary binary from the providied URL and make
it part of the project with `FetchContent_MakeAvailable`

#]=======================================================================]
function(rapids_cpm_download_proprietary_binary package_name url)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_download_proprietary_binary")

  # download and extract the binaries since they don't exist on the machine
  include(FetchContent)
  set(pkg_name "${package_name}_proprietary_binary")

  FetchContent_Declare(${pkg_name} URL ${url})
  FetchContent_MakeAvailable(${pkg_name})

  # Tell the subsequent rapids_cpm_find where to search so that it uses this binary
  set(${package_name}_ROOT "${${pkg_name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${package_name}_proprietary_binary ON PARENT_SCOPE)
endfunction()
