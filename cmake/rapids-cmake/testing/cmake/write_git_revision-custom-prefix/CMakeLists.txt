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
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.23.1)

project(DEMO VERSION 2.4 LANGUAGES CXX)
rapids_cmake_write_git_revision_file(demo_version "${CMAKE_CURRENT_BINARY_DIR}/demo/demo_git_version.hpp")
rapids_cmake_write_git_revision_file(nested_version "${CMAKE_CURRENT_BINARY_DIR}/nested/nested_git_version.h" PREFIX NESTED)

add_executable(write_git_version main.cpp)
target_link_libraries(write_git_version PRIVATE demo_version nested_version)
target_compile_features(write_git_version PRIVATE cxx_std_14)
