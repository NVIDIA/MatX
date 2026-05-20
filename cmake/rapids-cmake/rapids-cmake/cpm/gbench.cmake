# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_gbench
-----------------

.. versionadded:: v22.12.00

Allow projects to find or build Google Benchmark via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of Google benchmark :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_gbench( [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [BUILD_STATIC]
                     [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: benchmark
.. include:: common_package_args.txt

.. versionadded:: v23.12.00

``BUILD_STATIC``
  Will build Google Benchmark statically. No local searching for a previously
  built version will occur.

Result Targets
^^^^^^^^^^^^^^
  benchmark::benchmark targets will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`benchmark_SOURCE_DIR` is set to the path to the source directory of GBench.
  :cmake:variable:`benchmark_BINARY_DIR` is set to the path to the build directory of  GBench.
  :cmake:variable:`benchmark_ADDED`      is set to a true value if GBench has not been added before.
  :cmake:variable:`benchmark_VERSION`    is set to the version of GBench specified by the versions.json.
#]=======================================================================]
function(rapids_cpm_gbench)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.gbench")

  set(build_shared ON)
  if(BUILD_STATIC IN_LIST ARGN)
    set(build_shared OFF)
    set(CPM_DOWNLOAD_benchmark ON) # Since we need static we build from source
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(benchmark ${ARGN} VERSION_VAR version FIND_VAR find_args CPM_VAR
                          cpm_find_info TO_INSTALL_VAR to_install)

  include("${rapids-cmake-dir}/cmake/install_lib_dir.cmake")
  rapids_cmake_install_lib_dir(lib_dir)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(benchmark ${version} ${find_args}
                  GLOBAL_TARGETS benchmark::benchmark benchmark::benchmark_main
                  CPM_ARGS ${cpm_find_info}
                  OPTIONS "BENCHMARK_ENABLE_GTEST_TESTS OFF" "BENCHMARK_ENABLE_TESTING OFF"
                          "BENCHMARK_ENABLE_INSTALL ${to_install}"
                          "CMAKE_INSTALL_LIBDIR ${lib_dir}" "BUILD_SHARED_LIBS ${build_shared}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(benchmark)

  if(NOT TARGET benchmark::benchmark AND TARGET benchmark)
    add_library(benchmark::benchmark ALIAS benchmark)
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(benchmark_SOURCE_DIR "${benchmark_SOURCE_DIR}" PARENT_SCOPE)
  set(benchmark_BINARY_DIR "${benchmark_BINARY_DIR}" PARENT_SCOPE)
  set(benchmark_ADDED "${benchmark_ADDED}" PARENT_SCOPE)
  set(benchmark_VERSION ${version} PARENT_SCOPE)
endfunction()
