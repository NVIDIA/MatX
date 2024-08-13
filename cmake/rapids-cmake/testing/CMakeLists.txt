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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
project(rapids-cmake-testing LANGUAGES NONE)

enable_testing()
include(utils/cmake_config_test.cmake)
include(utils/cmake_ctest_test.cmake)
include(utils/cmake_build_test.cmake)

add_subdirectory(cmake)
add_subdirectory(cuda)
add_subdirectory(export)
add_subdirectory(find)

include("${CMAKE_CURRENT_SOURCE_DIR}/utils/determine_gpu_count.cmake")
determine_gpu_count()
add_subdirectory(test)

option(RAPIDS_CMAKE_ENABLE_DOWNLOAD_TESTS "Enable tests that require downloads" ON)
if(RAPIDS_CMAKE_ENABLE_DOWNLOAD_TESTS)
  # Setup and fill the cpm cache, and specify CPM_SOURCE_CACHE
  # so that all subsequent tests in this directory use the
  # shared CPM_SOURCE_CACHE
  include("${CMAKE_CURRENT_SOURCE_DIR}/utils/setup_cpm_cache.cmake")
  setup_cpm_cache()

  add_subdirectory(cpm)
  add_subdirectory(cython)
  add_subdirectory(cython-core)
  add_subdirectory(other)
endif()
