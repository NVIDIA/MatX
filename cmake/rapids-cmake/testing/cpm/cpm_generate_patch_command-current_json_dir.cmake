#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake)

rapids_cpm_init()
rapids_cpm_generate_patch_command(pkg_with_patch 10.2 patch_command)
if(patch_command)
  message(FATAL_ERROR "pkg_with_patch doesn't have override yet, patch_command should be empty")
endif()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "pkg_with_patch" : {
      "version" : "10.2",
      "git_tag" : "a_tag",
      "git_shallow" : "OFF",
      "exclude_from_all" : "ON",
      "patches" : [
        {
          "file" : "${current_json_dir}/example.diff",
          "issue" : "explain",
          "fixed_in" : ""
        },
        {
          "file" : "${current_json_dir}/example2.diff",
          "issue" : "explain",
          "fixed_in" : ""
        }
      ]
    }
  }
}
  ]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

rapids_cpm_generate_patch_command(pkg_with_patch 10.2 patch_command)
if(NOT patch_command)
  message(FATAL_ERROR "rapids_cpm_package_override specified a patch step for `pkg_with_patch`")
endif()

set(to_match_string "set(files \"${CMAKE_CURRENT_BINARY_DIR}/example.diff;${CMAKE_CURRENT_BINARY_DIR}/example2.diff\")")

list(POP_BACK patch_command script_to_run)
file(READ "${script_to_run}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_generate_patch_command failed to handle the `current_json_dir` placeholder")
endif()
