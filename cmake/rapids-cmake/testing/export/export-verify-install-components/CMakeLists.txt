# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/export.cmake)

cmake_minimum_required(VERSION 3.20)
project(fake LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

add_library(fakeLib_c1 INTERFACE)
install(TARGETS fakeLib_c1 EXPORT fake_c1_export)
include(${rapids-cmake-dir}/export/package.cmake)
rapids_export_package(INSTALL CUDAToolkit fake_c1_export)

add_library(fakeLib_c2 INTERFACE)
install(TARGETS fakeLib_c2 EXPORT fake_c2_exports)

add_library(fakeLib_c2_lib INTERFACE)
install(TARGETS fakeLib_c2_lib EXPORT fake_c2-lib_exports)

set(CMAKE_INSTALL_LIBDIR "lib64")
rapids_export(INSTALL fake
              EXPORT_SET fake_set
              COMPONENTS c1 c2 c2
              COMPONENTS_EXPORT_SET fake_c1_export fake_c2_exports fake_c2-lib_exports
              NAMESPACE test::)

# Add a custom command that install each component into a different temp location
add_custom_target(install_each_component_files ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --component fake --prefix
                          install/fake/
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --component c1 --prefix
                          install/c1/
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --component c2 --prefix
                          install/c2/)

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/install/CMakeLists.txt"
     [=[
cmake_minimum_required(VERSION 3.20)
project(verify_install_targets LANGUAGES CXX)

set(fake_files
  fake/lib64/cmake/fake/fake-config-version.cmake
  fake/lib64/cmake/fake/fake-config.cmake
  fake/lib64/cmake/fake/fake-targets.cmake
)
set(c1_files
  c1/lib64/cmake/fake/fake-c1-dependencies.cmake
  c1/lib64/cmake/fake/fake-c1-targets.cmake
)
set(c2_files
  c2/lib64/cmake/fake/fake-c2-targets.cmake
  c2/lib64/cmake/fake/fake-c2-lib-targets.cmake
)

foreach(file IN LISTS fake_files c1_files c2_files)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
    message(FATAL_ERROR "rapids_export(COMPONENTS) failed to install ${file}")
  endif()
endforeach()
]=])

add_custom_target(verify_install_components ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/install/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/install"
                          -B="${CMAKE_BINARY_DIR}/install/build")
add_dependencies(verify_install_components install_each_component_files)
