# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_nvbench
------------------

.. versionadded:: v21.10.00

Allow projects to find or build `nvbench` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of nvbench :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_nvbench( [BUILD_EXPORT_SET <export-name>]
                      [INSTALL_EXPORT_SET <export-name>]
                      [BUILD_STATIC]
                      [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: nvbench
.. include:: common_package_args.txt

.. versionadded:: v23.12.00

``BUILD_STATIC``
  Will build nvbench statically. No local searching for a previously
  built version will occur.

.. versionchanged:: v24.02.00

NVBench_ENABLE_CUPTI is set to OFF by default.

Result Targets
^^^^^^^^^^^^^^
  nvbench::nvbench target will be created

  nvbench::main target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`nvbench_SOURCE_DIR` is set to the path to the source directory of nvbench.
  :cmake:variable:`nvbench_BINARY_DIR` is set to the path to the build directory of  nvbench.
  :cmake:variable:`nvbench_ADDED`      is set to a true value if nvbench has not been added before.
  :cmake:variable:`nvbench_VERSION`    is set to the version of nvbench specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_nvbench)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.nvbench")

  set(build_shared ON)
  if(BUILD_STATIC IN_LIST ARGN)
    set(build_shared OFF)
    set(CPM_DOWNLOAD_nvbench ON) # Since we need static we build from source
    set(CPM_DOWNLOAD_fmt ON) # Make sure we don't link to a preexisting shared fmt
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(nvbench ${ARGN} VERSION_VAR version FIND_VAR find_args CPM_VAR
                          cpm_find_info TO_INSTALL_VAR to_install)

  # CUDA::nvml is an optional package and might not be installed ( aka conda )
  find_package(CUDAToolkit REQUIRED)
  set(nvbench_with_nvml "OFF")
  if(TARGET CUDA::nvml)
    set(nvbench_with_nvml "ON")
  endif()

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(nvbench ${version} ${find_args}
                  GLOBAL_TARGETS nvbench::nvbench nvbench::main
                  CPM_ARGS ${cpm_find_info}
                  OPTIONS "NVBench_ENABLE_NVML ${nvbench_with_nvml}"
                          "NVBench_ENABLE_CUPTI OFF"
                          "NVBench_ENABLE_EXAMPLES OFF"
                          "NVBench_ENABLE_TESTING OFF"
                          "NVBench_ENABLE_INSTALL_RULES ${to_install}"
                          "BUILD_SHARED_LIBS ${build_shared}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(nvbench)

  if(nvbench_ADDED AND TARGET nvbench)
    # nvcc incorrectly sees some loops in fmt as unreachable.
    target_compile_options(nvbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress 128>)
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(nvbench_SOURCE_DIR "${nvbench_SOURCE_DIR}" PARENT_SCOPE)
  set(nvbench_BINARY_DIR "${nvbench_BINARY_DIR}" PARENT_SCOPE)
  set(nvbench_ADDED "${nvbench_ADDED}" PARENT_SCOPE)
  set(nvbench_VERSION ${version} PARENT_SCOPE)

  # nvbench creates the correct namespace aliases
endfunction()
