#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)


rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

set(CMAKE_INSTALL_LIBDIR "lib64")
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON BUILD_EXPORT_SET nvcomp-targets INSTALL_EXPORT_SET nvcomp-targets)

if(NOT nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored nvcomp override file failed to get proprietary binary version")
endif()

# Check the contents of the nvcomp-targets-release.cmake file to ensure that
# every line containing "_IMPORT_PREFIX" also contains "lib64"
file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/_deps/nvcomp_proprietary_binary-src/lib64/cmake/nvcomp/nvcomp-targets-release.cmake" nvcomp_targets_release_contents)
foreach(line IN LISTS nvcomp_targets_release_contents)
  string(FIND "${line}" "_IMPORT_PREFIX" _IMPORT_PREFIX_INDEX)
  if(_IMPORT_PREFIX_INDEX EQUAL -1)
    continue()
  endif()
  string(FIND "${line}" "lib64" _LIB64_INDEX)
  if(_LIB64_INDEX EQUAL -1)
    message(FATAL_ERROR "nvcomp-targets-release.cmake file does not contain lib64")
  endif()
endforeach()

# Install and check the install directory.
add_custom_target(install_project ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix check_nvcomp_lib_dir/install/
  )

# Need to capture the install directory based on the binary dir of this project, not the subproject used for testing.
set(expected_install_dir "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/install/")

# Add a custom command that verifies that the expect files have
# been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.23.1)
project(verify_install_targets LANGUAGES CXX)

message(\"Checking for lib64 directory in ${expected_install_dir}\")
if (NOT EXISTS ${expected_install_dir}/lib64)
  message(FATAL_ERROR \"The lib64 directory didn't exist!\")
endif()

set(nvcomp_ROOT \"${expected_install_dir}/lib64/cmake/nvcomp\")
find_package(nvcomp REQUIRED)

file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/stub.cpp\" \" \")
add_library(uses_nvcomp SHARED stub.cpp)
target_link_libraries(uses_nvcomp PRIVATE nvcomp::nvcomp)

")

add_custom_target(verify_nvcomp_lib_dir ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir" -B="${CMAKE_BINARY_DIR}/install/build"
)
add_dependencies(verify_nvcomp_lib_dir install_project)
