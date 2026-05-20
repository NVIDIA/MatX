# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/package.cmake)
include(${rapids-cmake-dir}/export/detail/post_find_package_code.cmake)

rapids_export_package(BUILD FAKE_PACKAGE test_export_set VERSION 22.08)
rapids_export_post_find_package_code(BUILD FAKE_PACKAGE "set(a ON)" EXPORT_SET test_export_set)
rapids_export_post_find_package_code(
  bUILd FAKE_PACKAGE [=[
set(b ON)
set(c ON)]=] EXPORT_SET test_export_set)
rapids_export_post_find_package_code(build FAKE_PACKAGE "set(d ON);set(e ON)" EXPORT_SET
                                     test_export_set)

include(${rapids-cmake-dir}/export/write_dependencies.cmake)
rapids_export_write_dependencies(BUILD test_export_set
                                 "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake")

set(to_match
    [=[if(FAKE_PACKAGE_FOUND)_set(a ON)_set(b ON)_set(c ON)_set(d ON)_set(e ON)__endif()]=])

file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake" contents NEWLINE_CONSUME)
string(REPLACE "\n" "_" contents "${contents}")

string(FIND "${contents}" "${to_match}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(BUILD) failed to record rapids_export_post_find_package_code"
  )
endif()
