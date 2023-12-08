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
cmake_minimum_required(VERSION 3.23.1)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/thrust.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

set(CMAKE_INSTALL_LIBDIR "lib")

rapids_cpm_init()

rapids_cpm_thrust(NAMESPACE RapidsTest
                  GLOBAL_TARGETS RapidsTest::Thrust
                  INSTALL_EXPORT_SET example_export_set)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT example_export_set)
target_link_libraries(fakeLib INTERFACE RapidsTest::Thrust)

rapids_export(INSTALL fake
  EXPORT_SET example_export_set
  NAMESPACE test::
  )

# Install our project so we can verify `thrust_create_target` is called
# automatically in the export-set
add_custom_target(install_project ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix install/fake/
  )

# Add a custom command that verifies that the expect files have
# been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/install/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(verify_install_targets LANGUAGES CXX)

set(computed_path "${CMAKE_CURRENT_SOURCE_DIR}/fake/lib/cmake/fake/")
find_package(fake REQUIRED NO_DEFAULT_PATH HINTS ${computed_path})
if(NOT TARGET RapidsTest::Thrust)
  message(FATAL_ERROR "Failed to construct RapidsTest::Thrust target")
endif()
]=])

add_custom_target(verify_install_thrust_works ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/install/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/install" -B="${CMAKE_BINARY_DIR}/install/build"
)
add_dependencies(verify_install_thrust_works install_project)
