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
include(${rapids-cmake-dir}/cpm/init.cmake)

# Need to write out multiple default files
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/defaultsA.json
  [=[
{
  "packages": {
    "nvbench": {
      "version": "nvbench_version",
      "git_url": "nvbench_url",
      "git_tag": "nvbench_tag"
    }
  }
}
  ]=])

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/defaultsB.json
  [=[
{
  "packages": {
    "nvbench": {
      "version": "not_read_version",
      "git_url": "not_nvbench_url",
      "git_tag": "not_nvbench_tag"
    },
    "rmm": {
      "version": "rmm_version",
      "git_url": "rmm_url",
      "git_tag": "rmm_tag"
    },
  }
}
  ]=])

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/defaultsC.json
  [=[
{
  "packages": {
    "GTest": {
      "version": "GTest_version",
      "git_url": "GTest_url",
      "git_tag": "GTest_tag"
    }
  }
}
  ]=])

# Emulate multiple projects calling `rapids_cpm_init` with different default files
rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/defaultsA.json")
rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/defaultsB.json")
rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/defaultsC.json")

foreach(proj IN ITEMS rmm nvbench GTest)
  # Verify that multiple custom defaults work
  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(${proj} version repository tag shallow exclude)
  if(NOT version STREQUAL "${proj}_version")
    message(FATAL_ERROR "${proj} default version field was removed.")
  endif()
  if(NOT repository STREQUAL "${proj}_url")
    message(FATAL_ERROR "${proj} default repository field was removed.")
  endif()
  if(NOT tag STREQUAL "${proj}_tag")
    message(FATAL_ERROR "${proj} default tag field was removed.")
  endif()
  if(CPM_DOWNLOAD_ALL)
    message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be false since since we just specified a defaults version file'")
  endif()
  unset(CPM_DOWNLOAD_ALL)
endforeach()
