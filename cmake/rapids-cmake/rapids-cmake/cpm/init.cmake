# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_init
---------------

.. versionadded:: v21.06.00

Establish the `CPM` and preset package infrastructure for the project.

.. code-block:: cmake

  rapids_cpm_init( [CUSTOM_DEFAULT_VERSION_FILE <json_default_file_path>]
                   [OVERRIDE <json_override_file_path>]
                   [GENERATE_PINNED_VERSIONS]
                   )

The CPM module will be downloaded based on the state of :cmake:variable:`CPM_SOURCE_CACHE` and
:cmake:variable:`ENV{CPM_SOURCE_CACHE}`. This allows multiple nested projects to share the
same download of CPM. If those variables aren't set the file will be cached
in the build tree of the calling project

.. versionadded:: v24.06.00
  ```
  CUSTOM_DEFAULT_VERSION_FILE
  ```
  This is an advanced option that allows projects to specify a custom default ``versions.json`` file that will
  be used instead of the one packaged inside rapids-cmake. Since this allows you to specify a new default
  ``versions.json`` it must contain information for all the built-in rapids-cmake packages ( cccl, GTest, rmm, etc )
  that are used by your project.

  Using a built-in rapids-cmake package without an explicit entry in the custom default version file is undefined behavior.

  If multiple calls to ``rapids_cpm_init`` occur with different default version files being used,
  each version file will be loaded. If a project is listed in multiple default version files, the first
  file values will be used, and all later calls for that packaged will be ignored.  This "first to record, wins"
  approach is used to match ``FetchContent``, and allows parent projects to override child
  projects.

  The provided json file must follow the ``versions.json`` format, which is :ref:`documented here<cpm_version_format>`.

  If the variable :cmake:variable:`RAPIDS_CMAKE_CPM_DEFAULT_VERSION_FILE` is specified it will be used
  in all calls to ``rapids_cpm_init`` instead of any explicit `CUSTOM_DEFAULT_VERSION_FILE` file, or
  usage of the rapids-cmake default version.json file.

.. versionadded:: v21.10.00
  ``OVERRIDE``
  Allows projects to override the default values for any :cmake:command:`rapids_cpm_find`,
  :ref:`rapids_cpm_* <cpm_pre-configured_packages>`, `CPM <https://github.com/cpm-cmake/CPM.cmake>`_,
  and :cmake:module:`FetchContent() <cmake:module:FetchContent>` package. By providing a secondary
  file with extra`CPM` preset package information for the project.

  If multiple calls to ``rapids_cpm_init`` occur with different ``OVERRIDE`` files being used,
  each version file will be loaded. If a project is listed in multiple override files, the first
  file values will be used, and all later calls for that packaged will be ignored.  This "first to record, wins"
  approach is used to match ``FetchContent``, and allows parent projects to override child
  projects.

  The provided json file must follow the `versions.json` format, which is :ref:`documented here<cpm_version_format>`.

  If the override file doesn't specify a value or package entry the default version will be used.

  .. versionadded:: v24.06.00

  If the variable :cmake:variable:`RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` is specified it will be used
  in all calls to ``rapids_cpm_init`` no matter the arguments. Any existing
  ``rapids_cpm_init(OVERRIDE`` files will be ignored, and all other calls will be treated as if this file was specified
  as the override.

.. versionadded:: v24.04.00
  ```
  GENERATE_PINNED_VERSIONS
  ```
  Generates a json file with all `CPM` dependencies with pinned version values.
  This allows for reproducible builds using the exact same state.
  The pinning file will be located at `<CMAKE_BINARY_DIR>/rapids-cmake/pinned_versions.json`

  .. versionadded:: v24.06.00

  If the variable :cmake:variable:`RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE` exists it will be treated
  as if all calls to ``rapids_cpm_init`` are requesting generation of a pinned versions file.
  In addition to any existing explicit `GENERATE_PINNED_VERSIONS` files, the file path contained
  in :cmake:variable:`RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE` will be used as a destination to
  write the pinned versions.json content.

.. note::
  Must be called before any invocation of :cmake:command:`rapids_cpm_find`.

#]=======================================================================]
function(rapids_cpm_init)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.init")

  set(_rapids_options GENERATE_PINNED_VERSIONS)
  set(_rapids_one_value CUSTOM_DEFAULT_VERSION_FILE OVERRIDE)
  set(_rapids_multi_value)
  cmake_parse_arguments(_RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}" ${ARGN})

  include("${rapids-cmake-dir}/cpm/detail/load_preset_versions.cmake")

  if(_RAPIDS_CUSTOM_DEFAULT_VERSION_FILE)
    rapids_cpm_load_preset_versions(PRESET_FILE "${_RAPIDS_CUSTOM_DEFAULT_VERSION_FILE}")
  else()
    rapids_cpm_load_preset_versions()
  endif()

  if(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE)
    include("${rapids-cmake-dir}/cpm/package_override.cmake")
    rapids_cpm_package_override("${RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE}")
  elseif(_RAPIDS_OVERRIDE)
    include("${rapids-cmake-dir}/cpm/package_override.cmake")
    rapids_cpm_package_override("${_RAPIDS_OVERRIDE}")
  endif()

  if(_RAPIDS_GENERATE_PINNED_VERSIONS)
    include("${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake")
    rapids_cpm_generate_pinned_versions(
      OUTPUT "${CMAKE_BINARY_DIR}/rapids-cmake/pinned_versions.json")
  endif()

  if(DEFINED RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE)
    include("${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake")
    rapids_cpm_generate_pinned_versions(OUTPUT "${RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE}")
  endif()

  include("${rapids-cmake-dir}/cpm/detail/download.cmake")
  rapids_cpm_download()

  # Propagate up any modified local variables that CPM has changed.
  #
  # Push up the modified CMAKE_MODULE_PATh to allow `find_package` calls to find packages that CPM
  # already added.
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endfunction()
