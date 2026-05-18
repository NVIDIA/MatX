# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_bs_thread_pool
-------------------------

.. versionadded:: v24.08.00

Allow projects to find or build `thread-pool` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of `thread-pool` :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_bs_thread_pool( [BUILD_EXPORT_SET <export-name>]
                             [INSTALL_EXPORT_SET <export-name>]
                             [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: bs_thread_pool
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  `BS::thread_pool` target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`bs_thread_pool_SOURCE_DIR` is set to the path to the source directory of `thread-pool`.
  :cmake:variable:`bs_thread_pool_BINARY_DIR` is set to the path to the build directory of `thread-pool`.
  :cmake:variable:`bs_thread_pool_ADDED`      is set to a true value if `thread-pool` has not been added before.
  :cmake:variable:`bs_thread_pool_VERSION`    is set to the version of `thread-pool` specified by the versions.json.

#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_cpm_bs_thread_pool)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.bs_thread_pool")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(bs_thread_pool ${ARGN} VERSION_VAR version FIND_VAR find_args CPM_VAR
                          cpm_find_info TO_INSTALL_VAR to_install)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(bs_thread_pool ${version} ${find_args} GLOBAL_TARGETS rapids_bs_thread_pool
                  CPM_ARGS ${cpm_find_info} DOWNLOAD_ONLY ON)

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(bs_thread_pool)

  # Extract the major version value of bs_thread_pool to use with `rapids-export` to setup
  # compatibility rules
  include("${rapids-cmake-dir}/cmake/parse_version.cmake")
  rapids_cmake_parse_version(MAJOR ${version} ${version})

  # Set up install rules Need to be re-entrant safe so only call when `bs_thread_pool_ADDED`
  if(bs_thread_pool_ADDED)
    if(NOT TARGET rapids_bs_thread_pool)
      add_library(rapids_bs_thread_pool INTERFACE)
      target_include_directories(rapids_bs_thread_pool
                                 INTERFACE "$<BUILD_INTERFACE:${bs_thread_pool_SOURCE_DIR}/include>"
                                           "$<INSTALL_INTERFACE:include>")
      target_compile_definitions(rapids_bs_thread_pool INTERFACE "BS_THREAD_POOL_ENABLE_PAUSE=1")
      target_compile_features(rapids_bs_thread_pool INTERFACE cxx_std_17 cuda_std_17)
      set_property(TARGET rapids_bs_thread_pool PROPERTY EXPORT_NAME thread_pool)
      install(TARGETS rapids_bs_thread_pool EXPORT bs_thread_pool-targets)
    endif()
    if(_RAPIDS_BUILD_EXPORT_SET)
      include("${rapids-cmake-dir}/export/export.cmake")
      rapids_export(BUILD bs_thread_pool
                    VERSION ${version}
                    EXPORT_SET bs_thread_pool-targets
                    GLOBAL_TARGETS thread_pool
                    NAMESPACE BS::)
      include("${rapids-cmake-dir}/export/find_package_root.cmake")
      rapids_export_find_package_root(BUILD bs_thread_pool [=[${CMAKE_CURRENT_LIST_DIR}]=]
                                      EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
    endif()
    if(to_install)
      include(GNUInstallDirs)
      install(DIRECTORY "${bs_thread_pool_SOURCE_DIR}/include/"
              DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
      include("${rapids-cmake-dir}/export/export.cmake")
      rapids_export(INSTALL bs_thread_pool
                    VERSION ${version}
                    EXPORT_SET bs_thread_pool-targets
                    GLOBAL_TARGETS thread_pool
                    NAMESPACE BS::)
    endif()
  endif()

  if(NOT TARGET BS::thread_pool AND TARGET rapids_bs_thread_pool)
    add_library(BS::thread_pool ALIAS rapids_bs_thread_pool)
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(bs_thread_pool_SOURCE_DIR "${bs_thread_pool_SOURCE_DIR}" PARENT_SCOPE)
  set(bs_thread_pool_BINARY_DIR "${bs_thread_pool_BINARY_DIR}" PARENT_SCOPE)
  set(bs_thread_pool_ADDED "${bs_thread_pool_ADDED}" PARENT_SCOPE)
  set(bs_thread_pool_VERSION ${version} PARENT_SCOPE)
endfunction()
