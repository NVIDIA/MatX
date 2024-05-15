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
project(rapids-test-project LANGUAGES CXX CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

rapids_test_init()

find_package(CUDAToolkit REQUIRED)
add_executable(verify_docs main.cpp)
target_compile_features(verify_docs PRIVATE cxx_std_17)

target_include_directories(verify_docs PRIVATE "${rapids-cmake-dir}/../docs/cpp_code_snippets/")
target_link_libraries(verify_docs PRIVATE CUDA::cudart)

enable_testing()
add_test(NAME simple_test COMMAND verify_docs)
rapids_test_gpu_requirements(simple_test GPUS 1 PERCENT 25)
