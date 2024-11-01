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

cmake_path(GET rapids-cmake-dir PARENT_PATH rapids.cmake-location)
cmake_path(APPEND rapids.cmake-location "RAPIDS.cmake")

include("${rapids.cmake-location}")

if(NOT DEFINED rapids-cmake_SOURCE_DIR)
  message(FATAL_ERROR "expected variable rapids-cmake_SOURCE_DIR to exist")
endif()

# make sure we have fetched rapids-cmake with git
if(EXISTS "${rapids-cmake_SOURCE_DIR}/.git")
  message(FATAL_ERROR "expected rapids-cmake to be downloaded via a zip file")
endif()

if(NOT EXISTS "${rapids-cmake_SOURCE_DIR}/init.cmake")
  message(FATAL_ERROR "expected init.cmake to exist")
endif()
