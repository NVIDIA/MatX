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

project(write_language-nested-dirs LANGUAGES NONE)

# verify that enabling langues CUDA in A/B allows us to build targets
# that use CUDA
add_subdirectory(A)

add_executable(write_language-nested-dirs
               main.cu)
target_link_libraries(write_language-nested-dirs PRIVATE A B)
