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
cmake_minimum_required(VERSION 3.20)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

rapids_test_init()

find_package(CUDAToolkit REQUIRED)

add_executable(verify_multi_concurrent main.cu)
target_link_libraries(verify_multi_concurrent PRIVATE CUDA::cudart)
target_include_directories(verify_multi_concurrent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(verify_multi_concurrent PRIVATE "BINARY_DIR=${CMAKE_BINARY_DIR}")

enable_testing()

foreach(v RANGE 5)
  # this way all our tests can reside on the same gpu
  rapids_test_add(NAME concurrent_test_${v} COMMAND verify_multi_concurrent ${v} GPUS 1 PERCENT 1${v})
endforeach()
