#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN)
    set(to_install ON)
  endif()

  set(build_shared ON)
  if(BUILD_STATIC IN_LIST ARGN)
    set(build_shared OFF)
    set(CPM_DOWNLOAD_nvbench ON) # Since we need static we build from source
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(nvbench version repository tag shallow exclude)

  # CUDA::nvml is an optional package and might not be installed ( aka conda )
  find_package(CUDAToolkit REQUIRED)
  set(nvbench_with_nvml "OFF")
  if(TARGET CUDA::nvml)
    set(nvbench_with_nvml "ON")
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(nvbench ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(nvbench ${version} ${ARGN}
                  GLOBAL_TARGETS nvbench::nvbench nvbench::main
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "NVBench_ENABLE_NVML ${nvbench_with_nvml}"
                          "NVBench_ENABLE_CUPTI OFF"
                          "NVBench_ENABLE_EXAMPLES OFF"
                          "NVBench_ENABLE_TESTING OFF"
                          "NVBench_ENABLE_INSTALL_RULES ${to_install}"
                          "BUILD_SHARED_LIBS ${build_shared}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(nvbench)

  # Propagate up variables that CPMFindPackage provide
  set(nvbench_SOURCE_DIR "${nvbench_SOURCE_DIR}" PARENT_SCOPE)
  set(nvbench_BINARY_DIR "${nvbench_BINARY_DIR}" PARENT_SCOPE)
  set(nvbench_ADDED "${nvbench_ADDED}" PARENT_SCOPE)
  set(nvbench_VERSION ${version} PARENT_SCOPE)

  # nvbench creates the correct namespace aliases
endfunction()
