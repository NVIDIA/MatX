#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/export/export.cmake)

cmake_minimum_required(VERSION 3.23.1)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib1 INTERFACE)
add_library(fakeLib2 INTERFACE)
set_target_properties(fakeLib2 PROPERTIES EXPORT_NAME namespacefor2only)
install(TARGETS fakeLib1 fakeLib2 EXPORT fake_set)

rapids_export(BUILD FakEProJecT
  EXPORT_SET fake_set
  LANGUAGES CXX
)

# Add a custom command that generates a second project
# that verifies our target was exported with the correct
# namespace
file(WRITE "${CMAKE_BINARY_DIR}/verify/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23.1)
project(verify_build_targets LANGUAGES CXX)
message(STATUS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-targets.cmake")
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-targets.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../fakeproject-targets.cmake")
if(NOT TARGET fakeLib1)
  message(FATAL_ERROR "rapids_export failed to generate correct namespace targets")
endif()
if(NOT TARGET namespacefor2only)
  message(FATAL_ERROR "rapids_export failed to generate correct namespace targets")
endif()
  ]=])

add_custom_target(verify_target_file ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/verify/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/verify" -B="${CMAKE_BINARY_DIR}/verify/build"
  )
