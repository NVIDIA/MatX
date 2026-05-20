# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

project(rapids-existing-target LANGUAGES CXX)

include("${rapids-cmake-testing-dir}/cpm/make_fake_project_build_dir_with_config.cmake")

make_fake_project_build_dir_with_config(RapidsTestFind 2021.01.02 RapidsTestFindConfig.cmake
                                        RapidsTestFindConfigVersion.cmake)

rapids_cpm_init()

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/RapidsTestFind-build/")

add_library(RapidsTest::RapidsTest IMPORTED INTERFACE)

rapids_cpm_find(RapidsTestFind 2021.01.02 GLOBAL_TARGETS RapidsTest::RapidsTest
                INSTALL_EXPORT_SET setA)

rapids_cpm_find(RapidsTestFind 2021.01.02 GLOBAL_TARGETS RapidsTest::RapidsTest
                INSTALL_EXPORT_SET setB)

rapids_cpm_find(RapidsTestFind 2021.01.02 GLOBAL_TARGETS RapidsTest::RapidsTest
                BUILD_EXPORT_SET setB)

function(verify_dependency_file file_name to_match)
  file(STRINGS "${file_name}" text)
  set(package_dependency_found FALSE)

  foreach(line IN LISTS text)
    if(line MATCHES ${to_match})
      set(package_dependency_found TRUE)
      break()
    endif()
  endforeach()

  if(NOT package_dependency_found)
    message(FATAL_ERROR "${file_name} failed to export RapidsTestFind properly")
  endif()

endfunction()

set(file_path "${CMAKE_CURRENT_BINARY_DIR}/build_export_set.cmake")
file(REMOVE "${file_path}")
rapids_export_write_dependencies(build setB "${file_path}")
verify_dependency_file("${file_path}" "[=[NAME;RapidsTestFind;VERSION;2021.01.02]=]")

set(file_path "${CMAKE_CURRENT_BINARY_DIR}/install_export_setA.cmake")
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/install_export_setA.cmake")
rapids_export_write_dependencies(build setB "${file_path}")
verify_dependency_file("${file_path}" "[=[find_dependency(RapidsTestFind)]=]")

set(file_path "${CMAKE_CURRENT_BINARY_DIR}/install_export_setB.cmake")
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/install_export_setB.cmake")
rapids_export_write_dependencies(build setB "${file_path}")
verify_dependency_file("${file_path}" "[=[find_dependency(RapidsTestFind)]=]")
