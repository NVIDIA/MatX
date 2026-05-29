# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_cuco
---------------------

.. versionadded:: v22.08.00

Allow projects to find or build `cuCollections` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of cuCollections :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_cuco( [BUILD_EXPORT_SET <export-name>]
                   [INSTALL_EXPORT_SET <export-name>]
                   [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: cuco
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  cuco::cuco target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`cuco_SOURCE_DIR` is set to the path to the source directory of cuco.
  :cmake:variable:`cuco_BINARY_DIR` is set to the path to the build directory of cuco.
  :cmake:variable:`cuco_ADDED`      is set to a true value if cuco has not been added before.
  :cmake:variable:`cuco_VERSION`    is set to the version of cuco specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_cuco)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.cuco")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(cuco ${ARGN} VERSION_VAR version FIND_VAR find_args CPM_VAR cpm_find_info
                          TO_INSTALL_VAR to_install)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(cuco ${version} ${find_args} GLOBAL_TARGETS cuco::cuco CPM_ARGS ${cpm_find_info}
                  OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
                          "INSTALL_CUCO ${to_install}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(cuco)

  # Propagate up variables that CPMFindPackage provide
  set(cuco_SOURCE_DIR "${cuco_SOURCE_DIR}" PARENT_SCOPE)
  set(cuco_BINARY_DIR "${cuco_BINARY_DIR}" PARENT_SCOPE)
  set(cuco_ADDED "${cuco_ADDED}" PARENT_SCOPE)
  set(cuco_VERSION ${version} PARENT_SCOPE)

endfunction()
