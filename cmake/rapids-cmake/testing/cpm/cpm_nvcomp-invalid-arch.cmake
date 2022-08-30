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

rapids_cpm_init()

if(TARGET nvcomp::nvcomp)
  message(FATAL_ERROR "Expected nvcomp::nvcomp expected to not exist")
endif()

set(CMAKE_SYSTEM_PROCESSOR "i686") # Don't do this outside of tests
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON)

if(nvcomp_proprietary_binary)
  message(FATAL_ERROR "Shouldn't have found a pre-built version of nvcomp for a non-existent CMAKE_SYSTEM_PROCESSOR key")
endif()
