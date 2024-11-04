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

include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override(${CMAKE_CURRENT_SOURCE_DIR}/override.json)

include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(DOWNLOAD_ONLY ON)

# Verify that the two files that we inserted into the RMM source tree exist
# Which proves the patches in the override are properly applied
if(NOT EXISTS "${rmm_SOURCE_DIR}/git_file_1.txt")
  message(FATAL_ERROR "failed to apply rmm first patch")
endif()

if(NOT EXISTS "${rmm_SOURCE_DIR}/git_file_2.txt")
  message(FATAL_ERROR "failed to apply rmm second patch")
endif()

# Verify that the one file that we inserted into the cuco source tree exists.
if(NOT EXISTS "${cuco_SOURCE_DIR}/git_file_1.txt")
  message(FATAL_ERROR "failed to apply cucco first patch")
endif()

add_custom_target(verify_generated_pins ALL
  COMMAND ${CMAKE_COMMAND} "-S${CMAKE_SOURCE_DIR}/verify/" "-B${CMAKE_BINARY_DIR}/verify_build" "-Drapids-cmake-dir=${rapids-cmake-dir}" "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}"
)
