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
rapids_cpm_load_preset_versions
-------------------------------

.. versionadded:: v21.10.00

Establish the `CPM` preset package information for the project.

.. code-block:: cmake

  rapids_cpm_load_preset_versions()

.. note::
  Will be called by the first invocation of :cmake:command:`rapids_cpm_init` or :cmake:command:`rapids_cpm_<pkg>`.

#]=======================================================================]
function(rapids_cpm_load_preset_versions)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.load_preset_versions")

  # Check if we have been loaded before, if so early terminate
  get_property(already_loaded GLOBAL PROPERTY rapids_cpm_load_presets SET)
  if(already_loaded)
    return()
  endif()
  set_property(GLOBAL PROPERTY rapids_cpm_load_presets "ON")

  # Load our json files
  file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../versions.json" json_data)

  # Determine all the projects that exist in the json file
  string(JSON package_count LENGTH "${json_data}" packages)
  math(EXPR package_count "${package_count} - 1")

  # For each project cache the subset of the json for that project in a global property

  # cmake-lint: disable=E1120
  foreach(index RANGE ${package_count})
    string(JSON package_name MEMBER "${json_data}" packages ${index})
    string(JSON data GET "${json_data}" packages "${package_name}")
    set_property(GLOBAL PROPERTY rapids_cpm_${package_name}_json "${data}")
    set_property(GLOBAL PROPERTY rapids_cpm_${package_name}_json_file "${filepath}")
  endforeach()

endfunction()
