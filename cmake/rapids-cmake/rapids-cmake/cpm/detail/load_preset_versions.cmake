# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_load_preset_versions
-------------------------------

.. versionadded:: v21.10.00

Establish the `CPM` preset package information for the project.

.. code-block:: cmake

  rapids_cpm_load_preset_versions([PRESET_FILE version_file])

.. note::
  Will be called by the first invocation of :cmake:command:`rapids_cpm_init` or :cmake:command:`rapids_cpm_<pkg>`.

#]=======================================================================]
function(rapids_cpm_load_preset_versions)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.load_preset_versions")

  set(_rapids_options)
  set(_rapids_one_value PRESET_FILE)
  set(_rapids_multi_value)
  cmake_parse_arguments(_RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}" ${ARGN})

  set(_rapids_preset_version_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../versions.json")
  if(_RAPIDS_PRESET_FILE)
    set(_rapids_preset_version_file "${_RAPIDS_PRESET_FILE}")
  endif()
  if(DEFINED RAPIDS_CMAKE_CPM_DEFAULT_VERSION_FILE)
    set(_rapids_preset_version_file "${RAPIDS_CMAKE_CPM_DEFAULT_VERSION_FILE}")
  endif()

  if(NOT EXISTS "${_rapids_preset_version_file}")
    message(FATAL_ERROR "rapids_cpm can't load '${_rapids_preset_version_file}' to find package version information, verify it exists"
    )
  endif()

  # Check if we have been loaded before, if so early terminate
  get_property(already_loaded GLOBAL PROPERTY rapids_cpm_load_presets_${_rapids_preset_version_file}
               SET)
  if(already_loaded)
    return()
  endif()
  set_property(GLOBAL PROPERTY rapids_cpm_load_presets_${_rapids_preset_version_file} "ON")

  # Load our json file
  file(READ "${_rapids_preset_version_file}" json_data)

  # Determine all the projects that exist in the json file
  string(JSON package_count LENGTH "${json_data}" packages)
  math(EXPR package_count "${package_count} - 1")

  # For each project cache the subset of the json for that project in a global property

  # cmake-lint: disable=E1120
  foreach(index RANGE ${package_count})
    string(JSON package_name MEMBER "${json_data}" packages ${index})
    string(JSON data GET "${json_data}" packages "${package_name}")

    # Normalize the names all to lower case. This will allow us to better support overrides with
    # different package name casing
    string(TOLOWER "${package_name}" normalized_pkg_name)
    get_property(already_exists GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_json SET)
    if(already_exists)
      # Warn that we have duplicate entries in the default json file
      message(AUTHOR_WARNING "RAPIDS-CMake has detected two entries for ${package_name} in ${_rapids_preset_version_file}. Please ensure a single entry as all names are cased insensitive"
      )
    else()
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_json "${data}")
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_json_file
                                   "${_rapids_preset_version_file}")
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_proper_name "${package_name}")
    endif()
  endforeach()

endfunction()
