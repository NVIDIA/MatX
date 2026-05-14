# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
      "always_download" : true,
      "version": "3.0.6",
      "git_url": "https://github.com/NVIDIA/nvcomp.git",
      "git_tag": "v2.2.0",
      "proprietary_binary_cuda_version_mapping": {
        "11": "11",
        "12": "12",
        "13": "12"
      },
      "proprietary_binary": {
        "x86_64-linux": "https://developer.download.nvidia.com/compute/nvcomp/${version}/local_installers/nvcomp_${version}_x86_64_${cuda-toolkit-version-mapping}.x.tgz",
        "aarch64-linux": "https://developer.download.nvidia.com/compute/nvcomp/${version}/local_installers/nvcomp_${version}_SBSA_${cuda-toolkit-version-mapping}.x.tgz"
      }
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

#
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON)

if(NOT nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored nvcomp override file failed to get proprietary binary version")
endif()

if(EXISTS "${nvcomp_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Ignored nvcomp override file failed to get proprietary binary version")
endif()
