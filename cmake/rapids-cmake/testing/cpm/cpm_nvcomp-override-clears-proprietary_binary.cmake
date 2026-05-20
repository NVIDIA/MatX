# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp not to exist")
endif()

# Need to write out an nvcomp override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "nvcomp": {
      "version": "224",
      "git_url": "https://github.com/NVIDIA/nvcomp.git",
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

#
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON DOWNLOAD_ONLY ON)

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored nvcomp override file and brought in the binary version")
endif()

if(NOT EXISTS "${nvcomp_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Ignored nvcomp override file and brought in the binary version")
endif()
