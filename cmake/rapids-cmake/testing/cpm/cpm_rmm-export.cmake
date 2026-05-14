# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)

rapids_cpm_init()

rapids_cpm_rmm(BUILD_EXPORT_SET frank INSTALL_EXPORT_SET test)
rapids_cpm_rmm(INSTALL_EXPORT_SET test2)

get_target_property(packages rapids_export_install_test PACKAGE_NAMES)
if(NOT rmm IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_rmm failed to record rmm needs to be exported")
endif()

get_target_property(packages rapids_export_install_test2 PACKAGE_NAMES)
if(NOT rmm IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_rmm failed to record rmm needs to be exported")
endif()

get_target_property(packages rapids_export_build_frank PACKAGE_NAMES)
if(NOT rmm IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_rmm failed to record rmm needs to be exported")
endif()
