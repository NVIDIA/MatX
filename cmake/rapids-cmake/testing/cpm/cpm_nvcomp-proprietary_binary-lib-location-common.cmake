# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON BUILD_EXPORT_SET nvcomp-targets
                  INSTALL_EXPORT_SET nvcomp-targets)

if(NOT nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored nvcomp override file failed to get proprietary binary version")
endif()

# Check the contents of the nvcomp cmake files to ensure that every line containing "_IMPORT_PREFIX"
# also contains "${CMAKE_INSTALL_LIBDIR}"
set(nvcomp_list_of_target_files "nvcomp-targets-dynamic-release.cmake"
                                "nvcomp-targets-static-release.cmake")
if(nvcomp_VERSION VERSION_LESS "5.0")
  list(APPEND nvcomp_list_of_target_files "nvcomp-targets-common-release.cmake")
endif()
foreach(filename IN LISTS nvcomp_list_of_target_files)
  file(STRINGS
       "${CMAKE_CURRENT_BINARY_DIR}/_deps/nvcomp_proprietary_binary-src/${CMAKE_INSTALL_LIBDIR}/cmake/nvcomp/${filename}"
       nvcomp_targets_release_contents)
  foreach(line IN LISTS nvcomp_targets_release_contents)
    string(FIND "${line}" "_IMPORT_PREFIX" _IMPORT_PREFIX_INDEX)
    if(_IMPORT_PREFIX_INDEX EQUAL -1)
      continue()
    endif()
    cmake_path(GET CMAKE_INSTALL_LIBDIR FILENAME lib_dir_name)
    string(FIND "${line}" "${lib_dir_name}" _LIBDIR_INDEX)
    if(_LIBDIR_INDEX EQUAL -1)
      message(FATAL_ERROR "nvcomp-targets-release.cmake file does not contain ${lib_dir_name}")
    endif()
  endforeach()
endforeach()

# Install and check the install directory.
add_custom_target(install_project ALL COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
                                              --prefix check_nvcomp_lib_dir/install/)

# Need to capture the install directory based on the binary dir of this project, not the subproject
# used for testing.
set(expected_install_dir "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/install/")

# Add a custom command that verifies that the expect files have been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/CMakeLists.txt"
     "
cmake_minimum_required(VERSION 3.30.4)
project(verify_install_targets LANGUAGES CXX)

message(\"Checking for ${CMAKE_INSTALL_LIBDIR} directory in ${expected_install_dir}\")
if (NOT EXISTS ${expected_install_dir}/${CMAKE_INSTALL_LIBDIR})
  message(FATAL_ERROR \"The ${CMAKE_INSTALL_LIBDIR} directory didn't exist!\")
endif()

set(nvcomp_ROOT \"${expected_install_dir}/${CMAKE_INSTALL_LIBDIR}/cmake/nvcomp\")
find_package(nvcomp REQUIRED)

file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/stub.cpp\" \" \")
add_library(uses_nvcomp SHARED stub.cpp)
target_link_libraries(uses_nvcomp PRIVATE nvcomp::nvcomp)

")

add_custom_target(verify_nvcomp_lib_dir ALL
                  COMMAND ${CMAKE_COMMAND} -E rm -rf
                          "${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir/build"
                  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_nvcomp_lib_dir"
                          -B="${CMAKE_BINARY_DIR}/install/build")
add_dependencies(verify_nvcomp_lib_dir install_project)
