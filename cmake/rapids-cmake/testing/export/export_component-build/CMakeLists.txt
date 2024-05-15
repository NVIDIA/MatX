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
include(${rapids-cmake-dir}/export/detail/component.cmake)
include(${rapids-cmake-dir}/export/package.cmake)


cmake_minimum_required(VERSION 3.20)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export_package( build CUDAToolkit fake_set )

rapids_export_component(BUILD fakeproject component1 fake_set fake_set test::c1::)

# Add a custom command that generates a second project
# that verifies our component targets are exported correctly
file(WRITE "${CMAKE_BINARY_DIR}/verify-1/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(verify_build_targets LANGUAGES CXX)

message(STATUS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-targets.cmake")
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-targets.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct component config file")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-targets.cmake")
if(NOT TARGET test::c1::fakeLib)
  message(FATAL_ERROR "rapids_export failed to generate correct component target of test::c1::fakeLib")
endif()

message(STATUS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-dependencies.cmake")
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-dependencies.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct dependencies file")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../fakeproject-component1-dependencies.cmake")
if(NOT TARGET CUDA::toolkit)
  message(FATAL_ERROR "rapids_export failed to write a correct `CUDAToolkit` dependency")
endif()

]=])

add_custom_target(verify_target_component_files ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/verify-1/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/verify-1" -B="${CMAKE_BINARY_DIR}/verify-1/build"
  )
