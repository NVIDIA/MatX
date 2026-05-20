# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_cpm(INSTALL RMM test_set CPM_ARGS VERSION 2.0 FAKE_PACKAGE_ARGS FALSE
                  GLOBAL_TARGETS RMM::RMM_POOL)

rapids_export_cpm(INSTALL Thrust test_set CPM_ARGS VERSION 12.0 GLOBAL_TARGETS Thrust::Thrust)

rapids_export_cpm(INSTALL RMM test_set CPM_ARGS VERSION 2.0 FAKE_PACKAGE_ARGS FALSE
                  GLOBAL_TARGETS RMM::RMM)

rapids_export_write_dependencies(install test_set "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake")

# Parse the `export_set.cmake` file for correct number of `CPMFindPackage` calls and entries in
# `rapids_global_targets`

file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake" text)

set(find_package_count 0)
foreach(line IN LISTS text)
  # message(STATUS "1. line: ${line}")
  if(line MATCHES "CPMFindPackage")
    math(EXPR find_package_count "${find_package_count} + 1")
  endif()

  if(line MATCHES "set\\(rapids_global_targets" AND NOT line MATCHES "unset")
    # execute this line so we can check how many targets exist
    cmake_language(EVAL CODE "${line}")

    if(NOT "RMM::RMM" IN_LIST rapids_global_targets)
      message(FATAL_ERROR "Missing item [RMM::RMM] from list of targets to promote to global")
    endif()
    if(NOT "RMM::RMM_POOL" IN_LIST rapids_global_targets)
      message(FATAL_ERROR "Missing item [RMM::RMM_POOL] from list of targets to promote to global")
    endif()

    list(LENGTH rapids_global_targets orig_len)
    list(REMOVE_DUPLICATES rapids_global_targets)
    list(LENGTH rapids_global_targets uniquify_len)
    if(NOT orig_len EQUAL uniquify_len)
      message(FATAL_ERROR "Duplicate entries found in targets to promote to global")
    endif()
  endif()

endforeach()

if(NOT find_package_count EQUAL 2)
  message(FATAL_ERROR "Too many CPMFindPackage entries found. Expected 2, counted ${find_package_count}"
  )
endif()
