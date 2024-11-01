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
and all later calls for that packaged will be ignored.  This "first to record, wins"
approach is used to match FetchContent, and allows parent projects to override child
projects.

.. versionadded:: v24.06.00

If the variable :cmake:variable:`RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` is specified it will be used
in all calls to ``rapids_cpm_init``. Any existing explicit `OVERRIDE` files will be ignored, and
all other calls will be treated as if this file was specified as the override.


.. note::

  .. versionadded:: v23.10.00

    When the variable `CPM_<package_name>_SOURCE` exists, any override entries
    for `package_name` will be ignored.


.. note::
  If the override file doesn't specify a value or package entry the default
  version will be used.

  Must be called before any invocation of :cmake:command:`rapids_cpm_find`.

#]=======================================================================]
function(rapids_cpm_package_override _rapids_override_filepath)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_override")

  # The `RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` must be loaded instead of any explicit file path
  # when it is set
  if(DEFINED RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE)
    set(_rapids_override_filepath "${RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE}")
  endif()

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
      get_property(override_exists GLOBAL PROPERTY rapids_cpm_${package_name}_override_json DEFINED)
      if(NOT (override_exists OR DEFINED CPM_${package_name}_SOURCE))
        # only add the first override for a project we encounter
        string(JSON data GET "${json_data}" packages "${package_name}")
        set_property(GLOBAL PROPERTY rapids_cpm_${package_name}_override_json "${data}")
        set_property(GLOBAL PROPERTY rapids_cpm_${package_name}_override_json_file
                                     "${_rapids_override_filepath}")

        # establish the fetch content
        include(FetchContent)
        include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
        rapids_cpm_package_details(${package_name} version repository tag shallow exclude)

        include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
        rapids_cpm_generate_patch_command(${package_name} ${version} patch_command)

        unset(exclude_from_all)
        if(exclude AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.28.0)
          set(exclude_from_all EXCLUDE_FROM_ALL)
        endif()
        FetchContent_Declare(${package_name}
                             GIT_REPOSITORY ${repository}
                             GIT_TAG ${tag}
                             GIT_SHALLOW ${shallow}
                             ${patch_command} ${exclude_from_all})
      endif()
    endforeach()
  endif()
endfunction()
