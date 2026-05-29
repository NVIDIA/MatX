# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/package.cmake)

rapids_export_package(BUILD DifferingExportSets export1 GLOBAL_TARGETS EDT::EDT)
rapids_export_package(BUILD DifferingExportSets export2 GLOBAL_TARGETS EDT::EDT)

# Verify that we have the package and targets listed in both export sets
get_target_property(packages1 rapids_export_build_export1 PACKAGE_NAMES)
get_target_property(packages2 rapids_export_build_export2 PACKAGE_NAMES)

get_target_property(global_targets1 rapids_export_build_export1 GLOBAL_TARGETS)
get_target_property(global_targets2 rapids_export_build_export2 GLOBAL_TARGETS)

if(NOT packages1 STREQUAL packages2)
  message(FATAL_ERROR "rapids_export_package failed to record same package is in multiple export sets"
  )
endif()

if(NOT global_targets1 STREQUAL global_targets2)
  message(FATAL_ERROR "rapids_export_package failed to record same target is in multiple export sets"
  )
endif()
