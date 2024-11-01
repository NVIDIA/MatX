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
cmake_minimum_required(VERSION 3.26.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init(GENERATE_PINNED_VERSIONS)

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "rmm" : {
      "version" : "22.02",
      "git_tag" : "v22.02.00"
    }
  }
}
  ]=])
include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

add_custom_target(verify_generated_pins ALL
  COMMAND ${CMAKE_COMMAND} "-S${CMAKE_SOURCE_DIR}/verify/" "-B${CMAKE_BINARY_DIR}/verify_build" "-Drapids-cmake-dir=${rapids-cmake-dir}" "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}"
)
