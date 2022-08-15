#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
  message(FATAL_ERROR "Expected nvcomp::nvcomp expected to not exist")
endif()

# Need to write out an nvcomp override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages" : {
    "nvcomp" : {
      "version" : "224",
      "git_url" : "https://github.com/NVIDIA/nvcomp.git",
    }
  }
}
]=])
rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

#
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON)

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Ignored nvcomp override file and brought in the binary version")
endif()
