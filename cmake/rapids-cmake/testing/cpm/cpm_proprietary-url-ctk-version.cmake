#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/get_proprietary_binary_url.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)

rapids_cpm_init()

# Need to write out an override file with a proprietary blob url
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "test_binary" : {
      "version" : "2.6.1",
      "proprietary_binary" : {
        "x86_64-linux" :  "https://fake.url.com/${version}/${cuda-toolkit-version}/x86_64_${cuda-toolkit-version-major}.tgz",
        "aarch64-linux" : "https://fake.url.com/${version}/${cuda-toolkit-version}/aarch64_${cuda-toolkit-version-major}.tgz",
      }
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the placeholders are evaluated correctly from `find_package(CUDAToolkit)`
find_package(CUDAToolkit REQUIRED)
rapids_cpm_package_details(test_binary version repository tag shallow exclude)
rapids_cpm_get_proprietary_binary_url(test_binary ${version} url)

set(CTK_VER ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
set(CTK_VER_M ${CUDAToolkit_VERSION_MAJOR})
set(valid_url "https://fake.url.com/2.6.1/${CTK_VER}/${CMAKE_SYSTEM_PROCESSOR}_${CTK_VER_M}.tgz")
if(NOT valid_url STREQUAL url)
  message(FATAL_ERROR "Expected: ${valid_url} got: ${url}")
endif()
