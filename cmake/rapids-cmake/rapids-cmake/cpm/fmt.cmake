#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
rapids_cpm_fmt
-----------------

.. versionadded:: v23.04.00

Allow projects to find or build `fmt` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of fmt :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_fmt( [BUILD_EXPORT_SET <export-name>]
                  [INSTALL_EXPORT_SET <export-name>]
                  [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: fmt
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  fmt::fmt, fmt::fmt-header-only targets will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`fmt_SOURCE_DIR` is set to the path to the source directory of fmt.
  :cmake:variable:`fmt_BINARY_DIR` is set to the path to the build directory of fmt.
  :cmake:variable:`fmt_ADDED`      is set to a true value if fmt has not been added before.
  :cmake:variable:`fmt_VERSION`    is set to the version of fmt specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_fmt)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.fmt")

  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN)
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(fmt version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(fmt ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(fmt ${version} ${ARGN}
                  GLOBAL_TARGETS fmt::fmt fmt::fmt-header-only
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "FMT_INSTALL ${to_install}" "CMAKE_POSITION_INDEPENDENT_CODE ON")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(fmt)

  # Propagate up variables that CPMFindPackage provide
  set(fmt_SOURCE_DIR "${fmt_SOURCE_DIR}" PARENT_SCOPE)
  set(fmt_BINARY_DIR "${fmt_BINARY_DIR}" PARENT_SCOPE)
  set(fmt_ADDED "${fmt_ADDED}" PARENT_SCOPE)
  set(fmt_VERSION ${version} PARENT_SCOPE)

  # fmt creates the correct namespace aliases
endfunction()
