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

rapids_cpm_nvcomp()

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored no explicit enabling of `USE_PROPRIETARY_BINARY` and brought in the binary version"
  )
endif()

# Make sure we can be called multiple times
rapids_cpm_nvcomp()
