#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.21)
set(allowed_components A B C)
foreach(comp IN LISTS FakeDependency_FIND_COMPONENTS)
  if(${comp} IN_LIST allowed_components)
    set(FakeDependency_${comp}_FOUND ON)
    add_library(FakeDependency::${comp} INTERFACE IMPORTED)
  else()
    string(APPEND _FAIL_REASON "component '${comp}' was requested, but not found.  ")
  endif()
endforeach()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FakeDependency
    REASON_FAILURE_MESSAGE "${_FAIL_REASON}"
    HANDLE_COMPONENTS)
