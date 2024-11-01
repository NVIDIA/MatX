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
cmake_minimum_required(VERSION 3.26.4)
project(generate_resource_spec ${lang})

set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)

add_executable(generate_ctest_json generate_resource_spec.cpp)
if(cuda_toolkit)
  find_package(CUDAToolkit REQUIRED QUIET)
  target_link_libraries(generate_ctest_json PRIVATE CUDA::cudart_static)
  target_compile_definitions(generate_ctest_json PRIVATE HAVE_CUDA)
endif()
set_property(SOURCE generate_resource_spec.cpp PROPERTY LANGUAGE ${lang})

add_custom_command(TARGET generate_ctest_json POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:generate_ctest_json>
                           "${output_file}" COMMENT "Copying executable to destination")
