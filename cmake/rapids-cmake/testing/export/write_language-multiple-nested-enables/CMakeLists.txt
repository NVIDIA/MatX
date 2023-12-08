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
include(${rapids-cmake-dir}/export/write_language.cmake)

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(write_language-multiple-nested-enable LANGUAGES NONE)

# verify that enabling langues CXX && CUDA in A/B/D and A/C
# only installs one hook in `A` and `CMAKE_SOURCE_DIR`
add_subdirectory(A)

#Verify number of hooks is equal to 2
cmake_language(DEFER GET_CALL_IDS all_ids)
list(LENGTH all_ids list_len)
if(NOT list_len EQUAL 2)
  message(FATAL_ERROR "incorrect number of language hooks detected")
endif()

add_executable(write_language-multiple-nested-enable main.cpp)
target_link_libraries(write_language-multiple-nested-enable PRIVATE A B C D)
