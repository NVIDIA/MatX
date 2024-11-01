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
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)

rapids_cpm_init()
rapids_cpm_package_override("${CMAKE_CURRENT_BINARY_DIR}/../rapids-cmake/pinned_versions.json")

# Verify that each git_tag is now different.
rapids_cpm_package_details(rmm rmm_version rmm_repository pin_rmm_tag rmm_shallow rmm_exclude)
rapids_cpm_package_details(cuco cuco_version cuco_repository pin_cuco_tag cuco_shallow cuco_exclude)

if(NOT pin_rmm_tag STREQUAL "e3e32150d1bef13480ede4b2df1f0b38e43108a4")
  message(FATAL_ERROR "pinned rmm tag (${pin_rmm_tag}) doesn't match pre-computed SHA1 for the 22.02 tag")
endif()
if(NOT pin_cuco_tag STREQUAL "f823d30d6b08a60383266db25821074dbdbe5822")
  message(FATAL_ERROR "pinned cuco tag (${pin_cuco_tag}) doesn't match hardcoded SHA1(f823d30...)")
endif()
