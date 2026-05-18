# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/package.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_cpm(INSTALL RMM test_set CPM_ARGS RMM VERSION 2.0 FAKE_PACKAGE_ARGS FALSE
                  GLOBAL_TARGETS RMM::RMM_POOL)

rapids_export_package(install RMM test_set)
rapids_export_package(install ZLIB test_set)
rapids_export_package(install PNG test_set)
rapids_export_write_dependencies(install test_set "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake")

# Parse the `export_set.cmake` file for correct number of `CPMFindPackage` calls and entries in
# `rapids_global_targets`

file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake" text)

set(cpm_command_count 0)
set(find_dependency_command_count 0)
foreach(line IN LISTS text)
  # message(STATUS "1. line: ${line}")
  if(line MATCHES "CPMFindPackage")
    math(EXPR cpm_command_count "${cpm_command_count} + 1")
  elseif(line MATCHES "find_dependency")
    math(EXPR find_dependency_command_count "${find_dependency_command_count} + 1")
  endif()
endforeach()

if(NOT cpm_command_count EQUAL 1)
  message(FATAL_ERROR "Incorrect number of CPMFindPackage entries found. Expected 1, counted ${cpm_command_count}"
  )
endif()

if(NOT find_dependency_command_count EQUAL 2)
  message(FATAL_ERROR "Incorrect number of find_dependency entries found. Expected 2, counted ${find_dependency_command_count}"
  )
endif()
