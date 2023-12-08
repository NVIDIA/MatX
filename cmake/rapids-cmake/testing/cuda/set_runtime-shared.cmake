#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cuda/set_runtime.cmake)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp" " ")
add_library(uses_cuda SHARED ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
rapids_cuda_set_runtime(uses_cuda USE_STATIC FALSE)


get_target_property(runtime_state uses_cuda CUDA_RUNTIME_LIBRARY)
if( NOT runtime_state STREQUAL "SHARED")
  message(FATAL_ERROR "rapids_cuda_set_runtime didn't correctly set CUDA_RUNTIME_LIBRARY")
endif()

get_target_property(linked_libs uses_cuda LINK_LIBRARIES)
if(NOT "$<TARGET_NAME_IF_EXISTS:CUDA::cudart>" IN_LIST linked_libs)
  message(FATAL_ERROR "rapids_cuda_set_runtime didn't privately link to CUDA::cudart_static")
endif()
