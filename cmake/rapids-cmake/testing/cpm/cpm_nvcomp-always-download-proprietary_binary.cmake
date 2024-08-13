#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
      "proprietary_binary": {
        "x86_64-linux": "https://developer.download.nvidia.com/compute/nvcomp/${version}/local_installers/nvcomp_${version}_x86_64_${cuda-toolkit-version-major}.x.tgz",
        "aarch64-linux": "https://developer.download.nvidia.com/compute/nvcomp/${version}/local_installers/nvcomp_${version}_SBSA_${cuda-toolkit-version-major}.x.tgz"
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
