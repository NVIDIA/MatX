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
include(${rapids-cmake-dir}/cuda/set_architectures.cmake)

set(user_value "user-value")
set(CMAKE_CUDA_ARCHITECTURES ${user_value})
rapids_cuda_set_architectures(invalid-mode)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL user_value)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES shouldn't be modified by "
          "rapids_cuda_set_architectures() when past an invalid mode")
endif()
