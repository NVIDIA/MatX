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
      "git_url": "a_url",
      "git_tag": "a_tag",
      "proprietary_binary": {
        "x86_64-linux":  "https://fake.url.com/${version}/${cuda-toolkit-version}/x86_64_${cuda-toolkit-version-major}.tgz",
        "aarch64-linux": "https://fake.url.com/${version}/${cuda-toolkit-version}/aarch64_${cuda-toolkit-version-major}.tgz",
      }
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the placeholders are evaluated correctly from `enable_language(CUDA)`
rapids_cpm_package_info(test_binary VERSION_VAR version)
rapids_cpm_get_proprietary_binary_url(test_binary ${version} url)

find_package(CUDAToolkit)
set(CTK_VER ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
set(CTK_VER_M ${CUDAToolkit_VERSION_MAJOR})
set(valid_url "https://fake.url.com/2.6.1/${CTK_VER}/${CMAKE_SYSTEM_PROCESSOR}_${CTK_VER_M}.tgz")
if(NOT valid_url STREQUAL url)
  message(FATAL_ERROR "Expected: ${valid_url} got: ${url}")
endif()
