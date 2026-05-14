# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)

rapids_cpm_init()
set(CMAKE_CUDA_ARCHITECTURES OFF)
rapids_cpm_nvbench(BUILD_EXPORT_SET test)
rapids_cpm_nvbench(BUILD_EXPORT_SET test2 INSTALL_EXPORT_SET test2)

get_target_property(packages rapids_export_build_test PACKAGE_NAMES)
if(NOT nvbench IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_nvbench failed to record nvbench needs to be exported")
endif()

get_target_property(packages rapids_export_build_test2 PACKAGE_NAMES)
if(NOT nvbench IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_nvbench failed to record nvbench needs to be exported")
endif()

get_target_property(packages rapids_export_install_test2 PACKAGE_NAMES)
if(NOT nvbench IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_nvbench failed to record nvbench needs to be exported")
endif()
