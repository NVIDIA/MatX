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
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages": {
    "rmm": {
      "git_url": "new_rmm_url",
      "git_shallow": "OFF",
      "exclude_from_all": "ON"
    },
    "not_in_base": {
      "version": "1.0",
      "git_url": "new_rmm_url",
      "git_tag": "main",
      "git_shallow": "OFF",
      "exclude_from_all": "ON"
    }
  }
}
  ]=])

set(CPM_rmm_SOURCE "${CMAKE_CURRENT_BINARY_DIR}")
set(CPM_not_in_base_SOURCE "${CMAKE_CURRENT_BINARY_DIR}")
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override doesn't exist due to `CPM_rmm_SOURCE`
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

rapids_cpm_package_details(rmm version repository tag shallow exclude)
if(repository MATCHES "new_rmm_url")
  message(FATAL_ERROR "custom url field should not be set, due to CPM_rmm_SOURCE")
endif()
if(shallow MATCHES "OFF")
  message(FATAL_ERROR "shallow field should not be set, due to CPM_rmm_SOURCE")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should not be set, due to CPM_rmm_SOURCE")
endif()

unset(version)
unset(repository)
unset(tag)
rapids_cpm_package_details(not_in_base version repository tag shallow exclude)
if(version OR repository OR tag)
  message(FATAL_ERROR "rapids_cpm_package_details should not return anything for package that doesn't exist")
endif()
