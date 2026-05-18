# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)

# Ensure we use the cached download of nvcomp instead of downloading it again
if(EXISTS "${CPM_SOURCE_CACHE}/_deps/nvcomp_proprietary_binary-src/")
  set(nvcomp_ROOT "${CPM_SOURCE_CACHE}/_deps/nvcomp_proprietary_binary-src/")
endif()
rapids_cpm_init()
rapids_cpm_nvcomp(BUILD_EXPORT_SET test USE_PROPRIETARY_BINARY ON)
rapids_cpm_nvcomp(BUILD_EXPORT_SET test2 USE_PROPRIETARY_BINARY ON)

get_target_property(packages rapids_export_build_test PACKAGE_NAMES)
if(NOT nvcomp IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_nvcomp failed to record nvcomp needs to be exported")
endif()

get_target_property(packages rapids_export_build_test2 PACKAGE_NAMES)
if(NOT nvcomp IN_LIST packages)
  message(FATAL_ERROR "rapids_cpm_nvcomp failed to record nvcomp needs to be exported")
endif()
