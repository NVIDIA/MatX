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


set(CMAKE_CUDA_ARCHITECTURES "80")
rapids_cuda_init_architectures(rapids-project)
project(rapids-project LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES should exist after calling rapids_cuda_init_architectures()")
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80")
  message(FATAL_ERROR "rapids_cuda_init_architectures didn't ignore users CUDA_ARCHITECTURES value")
endif()
