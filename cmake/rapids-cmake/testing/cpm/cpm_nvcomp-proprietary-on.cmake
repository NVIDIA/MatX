# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)

rapids_cpm_init()

# Ensure we use the cached download of nvcomp instead of downloading it again
if(EXISTS "${CPM_SOURCE_CACHE}/_deps/nvcomp_proprietary_binary-src/")
  set(nvcomp_ROOT "${CPM_SOURCE_CACHE}/_deps/nvcomp_proprietary_binary-src/")
endif()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON)

if(NOT TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp target to exist")
endif()

# Make sure CUDA::cudart_static isn't in the interface link lines
get_target_property(libs nvcomp::nvcomp INTERFACE_LINK_LIBRARIES)

if("CUDA::cudart_static" IN_LIST libs)
  message(FATAL_ERROR "nvcomp::nvcomp shouldn't link to CUDA::cudart_static")
endif()

# Make sure we can be called multiple times
rapids_cpm_nvcomp()
