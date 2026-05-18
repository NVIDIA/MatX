# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_nvtx3
----------------

.. versionadded:: v24.06.00

Allow projects to find `nvtx3` via `CPM` with built-in tracking of dependencies
for correct export support.

Uses the version of nvtx3 :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_nvtx3( [BUILD_EXPORT_SET <export-name>]
                    [INSTALL_EXPORT_SET <export-name>]
                    [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: nvtx3
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  nvtx3::nvtx3-c, nvtx3::nvtx3-cpp targets will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`nvtx3_SOURCE_DIR` is set to the path to the source directory of nvtx3.
  :cmake:variable:`nvtx3_BINARY_DIR` is set to the path to the build directory of nvtx3.
  :cmake:variable:`nvtx3_ADDED`      is set to a true value if nvtx3 has not been added before.
  :cmake:variable:`nvtx3_VERSION`    is set to the version of nvtx3 specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_nvtx3)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.nvtx3")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(nvtx3 ${ARGN} VERSION_VAR version FIND_VAR find_args CPM_VAR
                          cpm_find_info TO_INSTALL_VAR to_install)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(nvtx3 ${version} ${find_args}
                  GLOBAL_TARGETS nvtx3-c nvtx3-cpp
                  CPM_ARGS ${cpm_find_info}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "NVTX3_INSTALL ON")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(nvtx3)

  # Propagate up variables that CPMFindPackage provide
  set(nvtx3_SOURCE_DIR "${nvtx3_SOURCE_DIR}" PARENT_SCOPE)
  set(nvtx3_BINARY_DIR "${nvtx3_BINARY_DIR}" PARENT_SCOPE)
  set(nvtx3_ADDED "${nvtx3_ADDED}" PARENT_SCOPE)
  set(nvtx3_VERSION ${version} PARENT_SCOPE)

endfunction()
