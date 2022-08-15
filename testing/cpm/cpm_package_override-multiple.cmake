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

# Load the default values for nvbench and GTest projects
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(nvbench nvbench_version nvbench_repository nvbench_tag nvbench_shallow nvbench_exclude)
rapids_cpm_package_details(GTest GTest_version GTest_repository GTest_tag GTest_shallow GTest_exclude)
rapids_cpm_package_details(rmm rmm_version rmm_repository rmm_tag rmm_shallow rmm_exclude)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override1.json
  [=[
{
  "packages" : {
    "nvbench" : {
      "git_tag" : "my_tag"
    },
    "GTest" : {
      "version" : "2.99"
    }
  }
}
  ]=])

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override2.json
  [=[
{
  "packages" : {
    "rmm" : {
      "git_tag" : "new_rmm_tag"
    },
    "GTest" : {
      "version" : "3.99"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override1.json)
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override2.json)

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

rapids_cpm_package_details(GTest version repository tag shallow exclude)
if(NOT version STREQUAL "3.99")
  message(FATAL_ERROR "custom version field was removed. ${version} was found instead")
endif()
if(NOT tag MATCHES "3.99")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead")
endif()

rapids_cpm_package_details(rmm version repository tag shallow exclude)
if(NOT tag MATCHES "new_rmm_tag")
  message(FATAL_ERROR "custom version field not used when computing git_tag value. ${tag} was found instead")
endif()
