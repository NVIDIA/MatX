# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_package_override
---------------------------

.. versionadded:: v21.10.00

Overrides the :cmake:command:`rapids_cpm_find`, :ref:`rapids_cpm_* <cpm_pre-configured_packages>`,
`CPM <https://github.com/cpm-cmake/CPM.cmake>`_, and :cmake:module:`FetchContent() <cmake:module:FetchContent>` package information for the project.

.. code-block:: cmake

  rapids_cpm_package_override(<json_file_path>)

Allows projects to override the default values for any :cmake:command:`rapids_cpm_find`,
:ref:`rapids_cpm_* <cpm_pre-configured_packages>`, `CPM <https://github.com/cpm-cmake/CPM.cmake>`_, and :cmake:module:`FetchContent() <cmake:module:FetchContent>` package.

The user provided json file must follow the `versions.json` format,
which is :ref:`documented here<cpm_version_format>`  and shown in the below
example:

.. literalinclude:: /packages/example_all_fields.json
  :language: json

By default when an override for a project is provided no local search
for that project will occur. This is done to make sure that the requested modified
version is used.

If a project is listed in multiple override files, the first file values will be used,
and all later calls for that package will be ignored.  This "first to record, wins"
approach is used to match FetchContent, and allows parent projects to override child
projects.

.. versionadded:: v24.06.00

If the variable :cmake:variable:`RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` is specified it will be used
in all calls to ``rapids_cpm_init`` no matter the arguments. Any existing
``rapids_cpm_init(OVERRIDE`` files will be ignored, and all other calls will be treated as if this file was specified
as the override.

.. note::

  .. versionadded:: v25.04.00

    When the variable `CPM_<package_name>_SOURCE` exists, any override entries
    for `package_name` will be parsed but ignored.

    For versions between v23.10 and v25.02 ( inclusive both sides ) the variable
    `CPM_<package_name>_SOURCE` will cause any override entries for `package_name`
    to be ignored and not parsed.


.. note::
  If the override file doesn't specify a value or package entry the default
  version will be used.

  Must be called before any invocation of :cmake:command:`rapids_cpm_find`.

#]=======================================================================]
function(rapids_cpm_package_override _rapids_override_filepath)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_override")

  if(NOT EXISTS "${_rapids_override_filepath}")
    message(FATAL_ERROR "rapids_cpm_package_override can't load '${_rapids_override_filepath}', verify it exists"
    )
  endif()
  file(READ "${_rapids_override_filepath}" json_data)

  # Determine all the projects that exist in the json file
  string(JSON package_count LENGTH "${json_data}" packages)
  math(EXPR package_count "${package_count} - 1")

  # For each project cache the subset of the json for that project in a global property so that
  # packasge_details.cmake can fetch that information
  if(package_count GREATER_EQUAL 0)
    # cmake-lint: disable=E1120
    foreach(index RANGE ${package_count})
      string(JSON package_name MEMBER "${json_data}" packages ${index})
      string(TOLOWER "${package_name}" normalized_pkg_name)
      get_property(override_exists GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_override_json
                   SET)
      if(override_exists)
        # Early terminate if this project already has an override
        continue()
      endif()

      # Warn if our name all lower case matches a default package, but our case sensitive names
      # doesn't ( ABC vs abc )
      get_property(package_proper_name GLOBAL
                   PROPERTY rapids_cpm_${normalized_pkg_name}_proper_name)
      if(package_proper_name AND NOT package_proper_name STREQUAL package_name)
        message(AUTHOR_WARNING "RAPIDS-CMake is assuming the override ${package_name} is meant for the ${package_proper_name} package. For correctness please use the correctly cased name"
        )
      endif()
      if(NOT package_proper_name)
        set(package_proper_name ${package_name}) # Required for FetchContent_Declare
      endif()

      # Always load overrides into our internal properties even when CPM_${package_name}_SOURCE is
      # set. We add the override_ignored so that `package_details` can maintain behavior around
      # `CPM_DOWNLOAD_ALL` when an override is loaded but not used
      #
      # We are using this behavior so that advanced users can retrieve the contents of an override
      # even when not in use
      string(JSON data GET "${json_data}" packages "${package_name}")
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_override_json "${data}")
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_override_json_file
                                   "${_rapids_override_filepath}")

      if(DEFINED CPM_${package_name}_SOURCE)
        set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_override_ignored "ON")
        continue()
      endif()
      set_property(GLOBAL PROPERTY rapids_cpm_${normalized_pkg_name}_override_ignored "OFF")

      # establish the fetch content
      include(FetchContent)
      include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
      rapids_cpm_package_info(${package_name} FOR_FETCH_CONTENT CPM_VAR cpm_find_info)

      message(DEBUG
              "rapids.cpm.rapids_cpm_package_override: FetchContent_Declare(${package_proper_name} ${cpm_find_info}) "
      )
      FetchContent_Declare(${package_proper_name} ${cpm_find_info})
      unset(package_proper_name)
    endforeach()
  endif()
endfunction()
