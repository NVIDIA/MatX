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
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.23.1)

project(DEMO VERSION 2.4 LANGUAGES CXX)

rapids_cmake_write_git_revision_file(git_generated_header "${CMAKE_CURRENT_BINARY_DIR}/git_version.hpp")

execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tag --dirty --always
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ERROR_QUIET
    OUTPUT_VARIABLE RAPIDS_WRITE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE #need to strip off any newline
    )

add_executable(is_dirty main.cpp)
target_link_libraries(is_dirty PRIVATE git_generated_header)
target_compile_features(is_dirty PRIVATE cxx_std_14)

if(RAPIDS_WRITE_VERSION MATCHES "dirty")
  target_compile_definitions(is_dirty PRIVATE "-DIS_DIRTY")
endif()
