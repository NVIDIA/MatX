#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
rapids_cpm_spdlog
-----------------

.. versionadded:: v21.10.00

Allow projects to find or build `spdlog` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of spdlog :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_spdlog( [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: spdlog
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  spdlog::spdlog, spdlog::spdlog_header_only targets will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`spdlog_SOURCE_DIR` is set to the path to the source directory of spdlog.
  :cmake:variable:`spdlog_BINARY_DIR` is set to the path to the build directory of  spdlog.
  :cmake:variable:`spdlog_ADDED`      is set to a true value if spdlog has not been added before.
  :cmake:variable:`spdlog_VERSION`    is set to the version of spdlog specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_spdlog)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.spdlog")

  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN)
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(spdlog version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(spdlog ${version} ${ARGN}
                  GLOBAL_TARGETS spdlog::spdlog spdlog::spdlog_header_only
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "SPDLOG_INSTALL ${to_install}")

  # Propagate up variables that CPMFindPackage provide
  set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
  set(spdlog_BINARY_DIR "${spdlog_BINARY_DIR}" PARENT_SCOPE)
  set(spdlog_ADDED "${spdlog_ADDED}" PARENT_SCOPE)
  set(spdlog_VERSION ${version} PARENT_SCOPE)

  # spdlog creates the correct namespace aliases
endfunction()
