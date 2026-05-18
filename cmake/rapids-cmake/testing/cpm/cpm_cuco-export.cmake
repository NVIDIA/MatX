# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cuco.cmake)

rapids_cpm_init()

rapids_cpm_cuco(BUILD_EXPORT_SET test_export_set)

get_target_property(packages rapids_export_build_test_export_set PACKAGE_NAMES)
if(NOT cuco IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_cuco failed to record cuco needs to be exported")
endif()
