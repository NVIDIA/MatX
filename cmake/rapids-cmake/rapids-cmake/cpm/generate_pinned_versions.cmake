# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_generate_pinned_versions
-----------------------------------

.. versionadded:: v24.04.00

Generate a json file with all dependencies with pinned version values

.. code-block:: cmake

  rapids_cpm_generate_pinned_versions( OUTPUT <json_verions_output_path> )

Generates a json file with all `CPM` dependencies with pinned version values.
This allows for subsequent reproducible builds using the exact same state.

The rapids-cmake default `versions.json` uses branch names or git tag names
for dependencies. This is done so that projects can 'live at head' of dependencies.
By using :cmake:command:`rapids_cpm_package_override` a project can specify a custom
`versions.json` that specifies exact git SHA's so that projects have reproducible builds.

:cmake:command:`rapids_cpm_generate_pinned_versions` can be used to transform a set of
rapids-cmake dependencies from branch names to pinned values. This can be used in subsequent
builds, e.g:

  1. Have CI run with `versions.json` which tracks dependency by branch name
  2. Store the generated pinned `versions.json` from the CI builds
  3. If build is good, create the release branch and commit the generated pinned `versions.json`
     to have reproducible builds for that release

``OUTPUT``
Specify a file path where the pinned versions information will be written. Can be called multiple
times and each unique path will be written to.

The generated json file will have the following entries for each package:

.. code-block:: json

  {
    "version": "<CPM_PACKAGE_<package_name>_VERSION>",
    "git_url": "<deduced>",
    "git_tag": "<deduced>",
    "git_shallow": false,
    "always_download": true
  }


If the original package (or override) also had any `patches`, or `proprietary_binary`
fields those will be propagated to the generated entry.

.. note::
  The git SHA1 computed for each package is found by finding the most recent
  commit that can be cloned from the url.

  This means that for proper reproducible builds, all patches must be encapsulated
  in the input json files or as CPM `PATCH_COMMAND`.

#]=======================================================================]
function(rapids_cpm_generate_pinned_versions)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.generate_pinned_versions")

  set(_rapids_options)
  set(_rapids_one_value OUTPUT)
  set(_rapids_multi_value)
  cmake_parse_arguments(_RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}" ${ARGN})

  if(NOT _RAPIDS_OUTPUT)
    message(FATAL_ERROR "rapids_cpm_generate_pinned_versions requires an `OUTPUT` argument")
  endif()

  find_package(Git QUIET)
  if(NOT Git_FOUND)
    message(FATAL_ERROR "rapids_cpm_generate_pinned_versions requires 'git' to exist")
  endif()

  # Append the requested write path for `detail/write_pinned_versions.cmake`
  set_property(GLOBAL APPEND PROPERTY rapids_cpm_generate_pin_files "${_RAPIDS_OUTPUT}")

  get_property(already_hooked GLOBAL PROPERTY rapids_cpm_generate_pin_hook SET)
  if(NOT already_hooked)
    # install a hook that writes out the pinned versions at the end of the root level CMakeLists.txt
    # execution so we get all CPM packages added. Plus we can compute the paths once, and write out
    # `N` times if needed
    set(root_dir "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}")
    cmake_language(DEFER DIRECTORY ${root_dir} ID rapids_cpm_generate_pinned_versions CALL include
                   "${rapids-cmake-dir}/cpm/detail/pinning_root_dir_hook.cmake")
    set_property(GLOBAL PROPERTY rapids_cpm_generate_pin_hook "ON")
  endif()

endfunction()
