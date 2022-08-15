#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/nvbench.cmake)

rapids_cpm_init()

if(TARGET nvbench::nvbench)
  message(FATAL_ERROR "Expected nvbench::nvbench expected to not exist")
endif()

rapids_cpm_nvbench()

set(targets_made nvbench::nvbench nvbench::main)

foreach(t IN LISTS targets_made)
  if(NOT TARGET ${t})
    message(FATAL_ERROR "Expected ${t} target to exist")
  endif()
endforeach()

# Make sure we can be called multiple times
rapids_cpm_nvbench()
