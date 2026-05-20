# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)

rapids_export_cpm(build FAKE_CPM_PACKAGE test_export_set CPM_ARGS FAKE_PACKAGE_ARGS TRUE)

rapids_export_cpm(BUILD SECOND_FAKE_CPM_PACKAGE test_export_set CPM_ARGS VERSION 2.0
                  GLOBAL_TARGETS ABC::ABC ABC::CBA)

if(NOT TARGET rapids_export_build_test_export_set)
  message(FATAL_ERROR "rapids_export_cpm failed to generate target for build")
endif()

# Verify that we encoded both packages for exporting
get_target_property(packages rapids_export_build_test_export_set PACKAGE_NAMES)
if(NOT FAKE_CPM_PACKAGE IN_LIST packages)
  message(FATAL_ERROR "rapids_export_cpm failed to record FAKE_CPM_PACKAGE needs to be exported")
endif()
if(NOT SECOND_FAKE_CPM_PACKAGE IN_LIST packages)
  message(FATAL_ERROR "rapids_export_cpm failed to record SECOND_FAKE_CPM_PACKAGE needs to be exported"
  )
endif()

# Verify that we encoded what `targets` are marked as global export
get_target_property(global_targets rapids_export_build_test_export_set GLOBAL_TARGETS)
if(NOT "ABC::ABC" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_cpm failed to record ABC::ABC needs to be global")
endif()
if(NOT "ABC::CBA" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_cpm failed to record ABC::CBA needs to be global")
endif()

# Verify that CPM property is set
get_target_property(requires_cpm rapids_export_build_test_export_set REQUIRES_CPM)
if(NOT requires_cpm)
  message(FATAL_ERROR "rapids_export_cpm failed to record that CPM is required by the export set")
endif()

# Verify that cpm configuration files exist
if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake"
   OR NOT EXISTS
      "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_SECOND_FAKE_CPM_PACKAGE.cmake")
  message(FATAL_ERROR "rapids_export_cpm failed to generate a CPM configuration")
endif()
