#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.23)
project(rapids-cpm-find-gtest-no-gmock LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mock_installed_gtest")

rapids_cpm_init()
rapids_cpm_gtest()

if (NOT TARGET GTest::gtest)
  message(FATAL_ERROR "rapids_cpm_find failed to add mocked gtest version")
endif()

if (TARGET GTest::gmock)
  message(FATAL_ERROR "rapids_cpm_find failed to add mocked gtest version")
endif()
