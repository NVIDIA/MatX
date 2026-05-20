# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/get_proprietary_binary_url.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_info.cmake)

rapids_cpm_init()

# Need to write out an override file with a proprietary blob url
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "test_binary": {
      "version": "2.6.1",
      "git_url": "empty",
      "git_tag": "empty",
      "proprietary_binary": {
        "x86_64-linux":  "https://fake.url.com/x86_${version}.tgz",
        "aarch64-linux": "https://fake.url.com/aarch_${version}.tgz",
      }
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

rapids_cpm_package_info(test_binary VERSION_VAR version)
rapids_cpm_get_proprietary_binary_url(test_binary ${version} nvcomp_url)

# Verify that we didn't go searching for the CUDAToolkit
if(TARGET CUDA::cudart_static OR TARGET CUDA::cudart)
  message(FATAL_ERROR "test_binary didn't use the cuda toolkit placeholder, but searching for it still occurred"
  )
endif()
