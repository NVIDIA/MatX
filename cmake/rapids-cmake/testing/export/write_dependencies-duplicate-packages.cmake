# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/package.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_package(BUILD ExactlyDuplicate test_set GLOBAL_TARGETS EDT::EDT)
rapids_export_package(BUILD DifferingPkgNameDuplicateTargets test_set GLOBAL_TARGETS PST::PST)
rapids_export_package(BUILD DifferingPkgNameV2DuplicateTargets test_set GLOBAL_TARGETS PST::PST)
rapids_export_package(build ExactlyDuplicate test_set GLOBAL_TARGETS EDT::EDT)

rapids_export_write_dependencies(build test_set "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake")

# Parse the `export_set.cmake` file for correct number of `find_dependency` calls and entries in
# `rapids_global_targets`

file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake" text)

foreach(line IN LISTS text)
  # message(STATUS "1. line: ${line}")
  if(line MATCHES "find_dependency\\(ExactlyDuplicate\\)")
    if(NOT DEFINED duplicate_package_parsed)
      set(duplicate_package_parsed TRUE)
    else()
      # We parsed a duplicate package
      message(FATAL_ERROR "Detected duplicate `find_dependency` calls for the same package")
    endif()
  endif()

  if(line MATCHES "set\\(rapids_global_targets" AND NOT line MATCHES "unset")
    # execute this line so we can check how many targets exist
    cmake_language(EVAL CODE "${line}")

    if(NOT "PST::PST" IN_LIST rapids_global_targets)
      message(FATAL_ERROR "Missing item [PST::PST] from list of targets to promote to global")
    endif()
    if(NOT "EDT::EDT" IN_LIST rapids_global_targets)
      message(FATAL_ERROR "Missing item [EDT::EDT] from list of targets to promote to global")
    endif()

    list(LENGTH rapids_global_targets orig_len)
    list(REMOVE_DUPLICATES rapids_global_targets)
    list(LENGTH rapids_global_targets uniquify_len)
    if(NOT orig_len EQUAL uniquify_len)
      message(FATAL_ERROR "Duplicate entries found in targets to promote to global")
    endif()
  endif()

endforeach()
