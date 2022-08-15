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

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(B)
add_subdirectory(C)

#Verify number of hooks is equal to 1
cmake_language(DEFER GET_CALL_IDS all_ids)

list(LENGTH all_ids list_len)
if(NOT list_len EQUAL 2)
  message(FATAL_ERROR "incorrect number of language hooks detected")
endif()

add_library(A STATIC static.cu)
