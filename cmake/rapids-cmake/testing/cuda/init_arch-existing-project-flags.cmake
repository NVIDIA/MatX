#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cuda/init_architectures.cmake)


# Verify that `RAPIDS` logic works correctly
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fileA.cmake" "set(file_A MAGIC_VALUE)")
set(CMAKE_PROJECT_example_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/fileA.cmake")

set(CMAKE_CUDA_ARCHITECTURES "RAPIDS")
rapids_cuda_init_architectures(example)

project(example LANGUAGES CUDA)
if(NOT DEFINED file_A)
  message(FATAL_ERROR "rapids_cuda_init_architectures can't overwrite existing `project()` include hooks")
endif()

# Verify that `NATIVE` logic works correctly
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fileB.cmake" "set(file_B MAGIC_VALUE)")
set(CMAKE_PROJECT_example2_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/fileB.cmake")

set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
rapids_cuda_init_architectures(example2)
project(example2 LANGUAGES CUDA)

if(NOT DEFINED file_B)
  message(FATAL_ERROR "rapids_cuda_init_architectures can't overwrite existing `project()` include hooks")
endif()
