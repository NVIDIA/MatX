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
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.23.1)
project(rapids-test-project LANGUAGES CXX)

include("${rapids-cmake-testing-dir}/cpm/make_fake_project_build_dir_with_config.cmake")

make_fake_project_build_dir_with_config(RapidsTestFind 2021.01.02
    RapidsTestFindConfig.cmake
    RapidsTestFindConfigVersion.cmake)

make_fake_project_build_dir_with_config(RapidsTestFind2 0.2
    rapidstestfind2-config.cmake
    rapidstestfind2-config-version.cmake)

rapids_cpm_init()

set(CMAKE_PREFIX_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/RapidsTestFind-build/"
  "${CMAKE_CURRENT_BINARY_DIR}/RapidsTestFind2-build/"
  )

rapids_cpm_find(RapidsTestFind 2021.01.02)
rapids_cpm_find(RapidsTestFind2 0.2)

if(NOT TARGET RapidsTest::RapidsTest)
  message(FATAL_ERROR "RapidsTest targets should be generated")
endif()
if(NOT TARGET RapidsTest2::RapidsTest)
  message(FATAL_ERROR "RapidsTest2 targets should be generated")
endif()
