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
rapids_cpm_spdlog
-----------------

.. versionadded:: v21.10.00

Allow projects to find or build `spdlog` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of spdlog :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_spdlog( [FMT_OPTION <fmt-option-name>]
                     [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [<CPM_ARGS> ...])

``FMT_OPTION``
.. versionadded:: v23.04.00

  Spdlog depends on the fmt library and offers multiple ways of handling this dependency when spdlog is built. This
  option only controls the behavior when spdlog is fetched and built, NOT when an installed spdlog is found on the
  system.

  This option can be set to: `BUNDLED`, `EXTERNAL_FMT`, `EXTERNAL_FMT_HO`, or `STD_FORMAT`. If set to
  `BUNDLED`, then spdlog will use its own bundled version of fmt. If set to `EXTERNAL_FMT` then spdlog will use the
  `fmt::fmt` target and be linked with the fmt library. If set to `EXTERNAL_FMT_HO` then spdlog will use the
  `fmt::fmt-header-only` target and be linked with a header only fmt library. If set to `STD_FORMAT` then spdlog
  will use `std::format` instead of the fmt library.

  Defaults to `EXTERNAL_FMT_HO`.

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
  :cmake:variable:`spdlog_fmt_target` is set to the fmt target used, if used

#]=======================================================================]
function(rapids_cpm_spdlog)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.spdlog")

  set(options)
  set(one_value FMT_OPTION BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find. Also
  # propagate the user provided build and install export sets.
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()
  if(_RAPIDS_BUILD_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS BUILD_EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
  endif()

  set(to_install OFF)
  if(_RAPIDS_INSTALL_EXPORT_SET)
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(spdlog version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(spdlog ${version} patch_command)

  # If the option wasn't passed to the command, default to header only fmt
  if(NOT _RAPIDS_FMT_OPTION)
    set(_RAPIDS_FMT_OPTION "EXTERNAL_FMT_HO")
  endif()

  if(_RAPIDS_FMT_OPTION STREQUAL "BUNDLED")
    set(spdlog_fmt_option "")
  elseif(_RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT")
    set(spdlog_fmt_option "SPDLOG_FMT_EXTERNAL ON")
    set(spdlog_fmt_target fmt::fmt)
  elseif(_RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT_HO")
    set(spdlog_fmt_option "SPDLOG_FMT_EXTERNAL_HO ON")
    set(spdlog_fmt_target fmt::fmt-header-only)
  elseif(_RAPIDS_FMT_OPTION STREQUAL "STD_FORMAT")
    set(spdlog_fmt_option "SPDLOG_USE_STD_FORMAT ON")
  else()
    message(FATAL_ERROR "Invalid option used for FMT_OPTION, got: ${_RAPIDS_FMT_OPTION}, expected one of: 'BUNDLED', 'EXTERNAL_FMT', 'EXTERNAL_FMT_HO', 'STD_FORMAT'"
    )
  endif()

  if(_RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT" OR _RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT_HO")
    include("${rapids-cmake-dir}/cpm/fmt.cmake")

    # Using `spdlog_ROOT` needs to cause any internal find calls in `spdlog-config.cmake` to first
    # search beside it before looking globally.
    list(APPEND fmt_ROOT ${spdlog_ROOT})

    rapids_cpm_fmt(${_RAPIDS_UNPARSED_ARGUMENTS})
  endif()

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(spdlog ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS spdlog::spdlog spdlog::spdlog_header_only
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "SPDLOG_INSTALL ${to_install}" "${spdlog_fmt_option}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(spdlog)

  # Propagate up variables that CPMFindPackage provide
  set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
  set(spdlog_BINARY_DIR "${spdlog_BINARY_DIR}" PARENT_SCOPE)
  set(spdlog_ADDED "${spdlog_ADDED}" PARENT_SCOPE)
  set(spdlog_VERSION ${version} PARENT_SCOPE)
  set(spdlog_fmt_target ${spdlog_fmt_target} PARENT_SCOPE)

  # spdlog creates the correct namespace aliases
endfunction()
