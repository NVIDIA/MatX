# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)

rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

rapids_cpm_nvcomp(DOWNLOAD_ONLY ON USE_PROPRIETARY_BINARY OFF)

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored `USE_PROPRIETARY_BINARY OFF` and brought in the binary version")
endif()

if(NOT EXISTS "${nvcomp_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Ignored USE_PROPRIETARY_BINARY OFF and brought in the binary version")
endif()

# Make sure we can be called multiple times
rapids_cpm_nvcomp()
