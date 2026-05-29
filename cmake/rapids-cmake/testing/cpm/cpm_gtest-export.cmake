# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

rapids_cpm_init()

rapids_cpm_gtest(BUILD_EXPORT_SET test)
rapids_cpm_gtest(BUILD_EXPORT_SET test2)

get_target_property(packages rapids_export_build_test PACKAGE_NAMES)
if(NOT GTest IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_gtest failed to record gtest needs to be exported")
endif()

get_target_property(packages rapids_export_build_test2 PACKAGE_NAMES)
if(NOT GTest IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_gtest failed to record gtest needs to be exported")
endif()
