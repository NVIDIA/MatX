#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.23.1)
project(rapids-test-project LANGUAGES CXX)


set(CPM_cucxx_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/mock_cucxx_source_dir")

rapids_cpm_init()
rapids_cpm_find(cucxx 1.0)

# Should find CPM generated `Findcucxx.cmake`
find_package(cucxx REQUIRED)

set(expected_find_path "${CPM_MODULE_PATH}/Findcucxx.cmake")
if(NOT EXISTS "${expected_find_path}")
  message(FATAL_ERROR "Findcucxx.cmake was not generated")
endif()

if(NOT TARGET MOCK_CUCXX)
  message(FATAL_ERROR "cucxx targets should be generated")
endif()
