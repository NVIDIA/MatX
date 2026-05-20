# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/package.cmake)

rapids_export_package(build FAKE_PACKAGE test_export_set COMPONENTS)

# Verify that package configuration files exist
set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_FAKE_PACKAGE.cmake")
if(NOT EXISTS "${path}")
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()

# verify that `COMPONENTS` isn't in FAKE_PACKAGE.cmake
set(to_match_string [=[COMPONENTS)]=])
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(NOT is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_package generated a find_package configuration with COMPONENTS"
  )
endif()
