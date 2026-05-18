# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)

rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

set(CMAKE_SYSTEM_PROCESSOR "i686") # Don't do this outside of tests
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON DOWNLOAD_ONLY ON)

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Shouldn't have found a pre-built version of nvcomp for a non-existent CMAKE_SYSTEM_PROCESSOR key"
  )
endif()
if(NOT EXISTS "${nvcomp_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Ignored USE_PROPRIETARY_BINARY OFF and brought in the binary version")
endif()
