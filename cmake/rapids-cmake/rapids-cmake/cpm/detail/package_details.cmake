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
rapids_cpm_package_details
--------------------------

. code-block:: cmake

  rapids_cpm_package_details(<package_name>
                             <version_variable>
                             <git_url_variable>
                             <git_tag_variable>
                             <shallow_variable>
                             <exclude_from_all_variable>
                             )

#]=======================================================================]
# cmake-lint: disable=R0913
function(rapids_cpm_package_details package_name version_var url_var tag_var shallow_var
         exclude_from_all_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_details")

  include("${rapids-cmake-dir}/cpm/detail/load_preset_versions.cmake")
  rapids_cpm_load_preset_versions()

  include("${rapids-cmake-dir}/cpm/detail/get_default_json.cmake")
  include("${rapids-cmake-dir}/cpm/detail/get_override_json.cmake")
  get_default_json(${package_name} json_data)
  get_override_json(${package_name} override_json_data)

  # Parse required fields
  function(rapids_cpm_json_get_value name)
    string(JSON value ERROR_VARIABLE have_error GET "${override_json_data}" ${name})
    if(have_error)
      string(JSON value ERROR_VARIABLE have_error GET "${json_data}" ${name})
    endif()

    if(NOT have_error)
      set(${name} ${value} PARENT_SCOPE)
    endif()
  endfunction()

  rapids_cpm_json_get_value(version)
  rapids_cpm_json_get_value(git_url)
  rapids_cpm_json_get_value(git_tag)

  # Parse optional fields, set the variable to the 'default' value first
  set(git_shallow ON)
  rapids_cpm_json_get_value(git_shallow)

  set(exclude_from_all OFF)
  rapids_cpm_json_get_value(exclude_from_all)

  if(override_json_data)
    # The default value for always download is defined by having an override for this package. When
    # we have an override we presume the user doesn't want to use an existing local installed
    # version
    set(always_download ON)
  endif()
  rapids_cpm_json_get_value(always_download)

  # Evaluate any magic placeholders in the version or tag components including the
  # `rapids-cmake-version` value
  if(NOT DEFINED rapids-cmake-version)
    include("${rapids-cmake-dir}/rapids-version.cmake")
  endif()

  cmake_language(EVAL CODE "set(version ${version})")
  cmake_language(EVAL CODE "set(git_tag ${git_tag})")

  set(${version_var} ${version} PARENT_SCOPE)
  set(${url_var} ${git_url} PARENT_SCOPE)
  set(${tag_var} ${git_tag} PARENT_SCOPE)
  set(${shallow_var} ${git_shallow} PARENT_SCOPE)
  set(${exclude_from_all_var} ${exclude_from_all} PARENT_SCOPE)
  if(DEFINED always_download)
    set(CPM_DOWNLOAD_ALL ${always_download} PARENT_SCOPE)
  endif()

endfunction()
