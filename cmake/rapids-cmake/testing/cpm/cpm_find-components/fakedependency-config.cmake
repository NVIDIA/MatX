# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.21)
set(allowed_components A B C)
foreach(comp IN LISTS FakeDependency_FIND_COMPONENTS)
  if(${comp} IN_LIST allowed_components)
    set(FakeDependency_${comp}_FOUND ON)
    add_library(FakeDependency::${comp} INTERFACE IMPORTED)
  else()
    string(APPEND _FAIL_REASON "component '${comp}' was requested, but not found.  ")
  endif()
endforeach()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FakeDependency REASON_FAILURE_MESSAGE "${_FAIL_REASON}"
                                  HANDLE_COMPONENTS)
