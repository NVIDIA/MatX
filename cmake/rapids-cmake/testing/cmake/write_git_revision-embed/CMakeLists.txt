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
rapids_cmake_write_git_revision_file(git_generated_header "${CMAKE_CURRENT_BINARY_DIR}/demo/git_version.hpp")

add_executable(embed_git_version main.cpp)
target_link_libraries(embed_git_version PRIVATE git_generated_header)
target_compile_features(embed_git_version PRIVATE cxx_std_14)

# verify that the git strings have been embedded.

find_package(Git QUIET)

add_custom_command(TARGET embed_git_version
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   -DWORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
                   -DEXECUTABLE=$<TARGET_FILE:embed_git_version>
                   -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_embedding.cmake)
