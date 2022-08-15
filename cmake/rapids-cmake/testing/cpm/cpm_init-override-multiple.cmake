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

rapids_cpm_init()

# Load the default values for nvbench and GTest projects
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(nvbench nvbench_version nvbench_repository nvbench_tag nvbench_shallow nvbench_exclude)
rapids_cpm_package_details(GTest GTest_version GTest_repository GTest_tag GTest_shallow GTest_exclude)


# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "nvbench" : {
      "git_tag" : "my_tag",
      "always_download" : false
    },
    "GTest" : {
      "version" : "2.99"
    }
  }
}
  ]=])

rapids_cpm_init(OVERRIDE "${CMAKE_CURRENT_BINARY_DIR}/override.json")

# Verify that the override works
rapids_cpm_package_details(nvbench version repository tag shallow exclude)
if(NOT version STREQUAL nvbench_version)
  message(FATAL_ERROR "default version field was removed.")
endif()
if(NOT repository STREQUAL nvbench_repository)
  message(FATAL_ERROR "default repository field was removed.")
endif()
if(NOT tag STREQUAL "my_tag")
  message(FATAL_ERROR "custom git_tag field was ignored. ${tag} found instead of my_url")
endif()
if(CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be false since the nvbench override explicitly sets it to 'false'")
endif()

rapids_cpm_package_details(GTest version repository tag shallow exclude)
if(NOT version STREQUAL "2.99")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "2.99")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead")
endif()
if(NOT CPM_DOWNLOAD_ALL)
  message(FATAL_ERROR "CPM_DOWNLOAD_ALL should be enabled by default when an override exists")
endif()
