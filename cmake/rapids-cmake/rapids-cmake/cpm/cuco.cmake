#=============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

  set(options)
  set(one_value INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have INSTALL_EXPORT_SET as this is need for rapids_cpm_find
  set(to_install OFF)
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(cuco version repository tag shallow exclude)

  set(to_exclude OFF)
  if(NOT to_install OR exclude)
    set(to_exclude ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(cuco ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(cuco ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS cuco::cuco
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${to_exclude}
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
