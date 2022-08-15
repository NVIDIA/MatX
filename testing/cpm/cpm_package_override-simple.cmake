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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "rmm" : {
      "git_tag" : "new_rmm_tag",
      "git_shallow" : "OFF",
      "exclude_from_all" : "ON"
    },
    "GTest" : {
      "version" : "3.00.A1"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

rapids_cpm_package_details(GTest version repository tag shallow exclude)
if(NOT version STREQUAL "3.00.A1")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "3.00.A1")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead")
endif()
if(NOT exclude MATCHES "OFF")
  message(FATAL_ERROR "default value of exclude not found. ${exclude} was found instead")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be set to true when an override exists")
endif()

rapids_cpm_package_details(rmm version repository tag shallow exclude)
if(NOT tag MATCHES "new_rmm_tag")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead")
endif()
if(NOT shallow MATCHES "OFF")
  message(FATAL_ERROR "override should not change git_shallow value. ${shallow} was found instead")
endif()
if(NOT exclude MATCHES "ON")
  message(FATAL_ERROR "override should have changed exclude value. ${exclude} was found instead")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be set to true when an override exists")
endif()
