#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages": {
    "pkg_with_patch": {
      "version": "10.2",
      "git_url" : "a_url",
      "git_tag": "a_tag",
      "git_shallow": "OFF",
      "exclude_from_all": "ON",
      "patches": [
        {
          "file": "e/example.diff",
          "issue": "explain",
          "fixed_in": ""
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

check_copyright_header("${CMAKE_BINARY_DIR}/rapids-cmake/patches/pkg_with_patch/patch.cmake")
