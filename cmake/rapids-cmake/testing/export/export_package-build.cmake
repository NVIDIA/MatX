# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/package.cmake)

rapids_export_package(BUILD ZLIB test_export_set GLOBAL_TARGETS ZLIB::ZLIB)
rapids_export_package(build PNG test_export_set)

if(NOT TARGET rapids_export_build_test_export_set)
  message(FATAL_ERROR "rapids_export_package failed to generate target for build")
endif()

# Verify that we encoded both packages for exporting
get_target_property(packages rapids_export_build_test_export_set PACKAGE_NAMES)
if(NOT ZLIB IN_LIST packages)
  message(FATAL_ERROR "rapids_export_package failed to record ZLIB needs to be exported")
endif()
if(NOT PNG IN_LIST packages)
  message(FATAL_ERROR "rapids_export_package failed to record PNG needs to be exported")
endif()

# Verify that we encoded what `targets` are marked as global export
get_target_property(global_targets rapids_export_build_test_export_set GLOBAL_TARGETS)
if(NOT "ZLIB::ZLIB" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_package failed to record ZLIB::ZLIB needs to be global")
endif()

# Verify that temp package configuration files exist
if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_ZLIB.cmake"
   OR NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_PNG.cmake")
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()
