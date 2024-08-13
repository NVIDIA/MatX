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
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Need to write out an override file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
  [=[
{
  "packages": {
    "rmm": {
      "patches": [
        {
        }
      ]
    },
    "CCCL": {
      "patches": [ ]
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# Verify that the override works
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")

rapids_cpm_package_details(rmm version repository tag shallow exclude)
rapids_cpm_generate_patch_command(rmm ${version} patch_command)
if(patch_command)
  message(FATAL_ERROR "no patch command expected for rmm")
endif()

rapids_cpm_package_details(CCCL version repository tag shallow exclude)
rapids_cpm_generate_patch_command(CCCL ${version} patch_command)
if(patch_command)
  message(FATAL_ERROR "no patch command expected for cccl")
endif()
