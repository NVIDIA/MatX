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
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages": {
    "custom_package_never_in_rapids" : {
      "version" : "3.1.0",
      "git_url" : "https://github.com/NVIDIA/NVTX",
      "git_tag" : "96aeb0d8702981972fac0f6e485fea7acbfd5446",
      "git_shallow" : false,
      "exclude_from_all" : true
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

rapids_cpm_package_details(custom_package_never_in_rapids version repository tag shallow exclude)
if(NOT version STREQUAL "3.1.0")
  message(FATAL_ERROR "expected version field wasn't found. ${version} was found instead")
endif()

# Make sure we can clone without getting an error when
# parsing the EXCLUDE_FROM_ALL tag
rapids_cpm_find(custom_package_never_in_rapids 3.1)
