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
rapids_cpm_rmm
--------------

.. versionadded:: v21.10.00

Allow projects to find or build `RMM` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the current rapids-cmake version of RMM `as specified in the version file <cpm_versions>`
for  consistency across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_rmm( [BUILD_EXPORT_SET <export-name>]
                  [INSTALL_EXPORT_SET <export-name>]
                  [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: rmm
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  rmm::rmm target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`rmm_SOURCE_DIR` is set to the path to the source directory of RMM.
  :cmake:variable:`rmm_BINARY_DIR` is set to the path to the build directory of  RMM.
  :cmake:variable:`rmm_ADDED`      is set to a true value if RMM has not been added before.
  :cmake:variable:`rmm_VERSION`    is set to the version of RMM specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_rmm)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rmm")

  set(options)
  set(one_value INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(rmm version repository tag shallow exclude)
  set(to_exclude OFF)
  if(NOT _RAPIDS_INSTALL_EXPORT_SET OR exclude)
    set(to_exclude ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(rmm ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(rmm ${version} ${ARGN} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS rmm::rmm
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${to_exclude}
                  OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(rmm)

  # Propagate up variables that CPMFindPackage provide
  set(rmm_SOURCE_DIR "${rmm_SOURCE_DIR}" PARENT_SCOPE)
  set(rmm_BINARY_DIR "${rmm_BINARY_DIR}" PARENT_SCOPE)
  set(rmm_ADDED "${rmm_ADDED}" PARENT_SCOPE)
  set(rmm_VERSION ${version} PARENT_SCOPE)

  # rmm creates the correct namespace aliases
endfunction()
