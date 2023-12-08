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
cmake_minimum_required(VERSION 3.20)

include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

project(fake LANGUAGES CXX VERSION 3.1.4)

rapids_cpm_init()

set(CMAKE_INSTALL_LIBDIR "lib/aarch64")
rapids_cpm_libcudacxx(INSTALL_EXPORT_SET fake_set)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)
target_link_libraries(fakeLib INTERFACE libcudacxx::libcudacxx)

rapids_export(INSTALL fake
  EXPORT_SET fake_set
  NAMESPACE test::
  )

# Install our project so we can verify `libcudacxx` can be found
# from a custom install location
add_custom_target(install_project ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix install/fake/
  )

# Add a custom command that verifies that the expect files have
# been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/install/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(verify_install_targets LANGUAGES CXX)

set(computed_path "${CMAKE_CURRENT_SOURCE_DIR}/fake/lib/aarch64/cmake/fake/")
find_package(fake REQUIRED NO_DEFAULT_PATH HINTS ${computed_path})
if(NOT TARGET libcudacxx::libcudacxx)
  message(FATAL_ERROR "Failed to import libcudacxx dependency")
endif()
]=])

add_custom_target(verify_install_libcudacxx_works ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/install/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/install" -B="${CMAKE_BINARY_DIR}/install/build"
)
add_dependencies(verify_install_libcudacxx_works install_project)
