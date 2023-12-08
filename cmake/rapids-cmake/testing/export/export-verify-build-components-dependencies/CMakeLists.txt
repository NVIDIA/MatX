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

cmake_minimum_required(VERSION 3.20)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

add_library(fakeLib_c1 INTERFACE)
install(TARGETS fakeLib_c1 EXPORT fake_set_c1_export)

include(${rapids-cmake-dir}/export/package.cmake)
rapids_export_package( build CUDAToolkit fake_set_c1_export)

rapids_export(BUILD FakEProJecT
  EXPORT_SET fake_set
  COMPONENTS c1
  COMPONENTS_EXPORT_SET fake_set_c1_export
  LANGUAGES CXX
  NAMESPACE test::
  )

# Add a custom command that generates a second project
# that verifies our component targets are exported correctly
file(WRITE "${CMAKE_BINARY_DIR}/verify-1/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(verify_build_targets LANGUAGES CXX)
set(nice_names c1-fake_set)
foreach(nice_name IN ZIP_LISTS components export_sets)
  include("${CMAKE_CURRENT_LIST_DIR}/../fakeproject-${nice_names}-dependencies.cmake")
  if(NOT TARGET CUDA::toolkit)
    message(FATAL_ERROR "rapids_export failed to generate correct dependencies imports[fakeproject-${nice_names}-dependencies.cmake]")
  endif()
endforeach()
]=])

add_custom_target(verify_target_dependencies_files ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/verify-1/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/verify-1" -B="${CMAKE_BINARY_DIR}/verify-1/build"
  )

# Add a custom command that generates a second project
# that verifies our component targets are exported correctly
file(WRITE "${CMAKE_BINARY_DIR}/verify-2/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(verify_build_targets LANGUAGES CXX)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)
find_package(fakeproject COMPONENTS c1 REQUIRED)
if(NOT TARGET CUDA::toolkit)
  message(FATAL_ERROR "rapids_export failed to generate correct dependencies imports")
endif()
]=])

add_custom_target(verify_target_dependencies_find ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/verify-2/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/verify-2" -B="${CMAKE_BINARY_DIR}/verify-2/build"
  )
