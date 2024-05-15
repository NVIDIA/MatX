#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/gbench.cmake)

rapids_cpm_init()


if(TARGET benchmark::benchmark)
  message(FATAL_ERROR "Expected benchmark::benchmark not to exist")
endif()

rapids_cpm_gbench()

if(NOT TARGET benchmark::benchmark)
  message(FATAL_ERROR "Expected benchmark::benchmark target to exist")
endif()

# Make sure we can be called multiple times
rapids_cpm_gbench()
