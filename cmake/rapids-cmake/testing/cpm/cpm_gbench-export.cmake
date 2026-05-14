# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gbench.cmake)

rapids_cpm_init()

rapids_cpm_gbench(BUILD_EXPORT_SET bench)
rapids_cpm_gbench(BUILD_EXPORT_SET bench2)

get_target_property(packages rapids_export_build_bench PACKAGE_NAMES)
if(NOT benchmark IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_gbench failed to record benchmark needs to be exported")
endif()

get_target_property(packages rapids_export_build_bench2 PACKAGE_NAMES)
if(NOT benchmark IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_gbench failed to record benchmark needs to be exported")
endif()
