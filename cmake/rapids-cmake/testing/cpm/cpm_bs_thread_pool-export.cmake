# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

rapids_cpm_init()

rapids_cpm_bs_thread_pool(BUILD_EXPORT_SET test)
rapids_cpm_bs_thread_pool(INSTALL_EXPORT_SET test2)

get_target_property(packages rapids_export_build_test PACKAGE_NAMES)
if(NOT bs_thread_pool IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_bs_thread_pool failed to record bs_thread_pool needs to be exported"
  )
endif()

get_target_property(packages rapids_export_install_test2 PACKAGE_NAMES)
if(NOT bs_thread_pool IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_bs_thread_pool failed to record bs_thread_pool needs to be exported"
  )
endif()
