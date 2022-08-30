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
include(${rapids-cmake-dir}/find/package.cmake)


function(track_normal_find_package count_var)
  # We need to establish CMAKE_MESSAGE_CONTEXT before we count
  # as it is expected to leak
  set(CMAKE_MESSAGE_CONTEXT "test")

  find_package(${ARGN})

  get_cmake_property(_all_local_variables VARIABLES)
  list(LENGTH _all_local_variables count_var)
  set(count_var ${count_var} PARENT_SCOPE)
endfunction()

function(track_rapids_find_package count_var)
  # We need to establish CMAKE_MESSAGE_CONTEXT before we count
  # as it is expected to leak
  set(CMAKE_MESSAGE_CONTEXT "test")
  rapids_find_package(${ARGN})

  get_cmake_property(_all_local_variables VARIABLES)
  list(LENGTH _all_local_variables count_var)
  set(count_var ${count_var} PARENT_SCOPE)

  # verify CMAKE_MESSAGE_CONTEXT has been properly popped
  list(LENGTH CMAKE_MESSAGE_CONTEXT context_len)
  if(context_len GREATER 1)
    message(FATAL_ERROR "CMAKE_MESSAGE_CONTEXT hasn't been properly reset")
  endif()
endfunction()


# Need to create both of the length variables ahead of time so
# that they are included in the counts and track_rapids_find_package
set(normal_len 0)
set(rapids_len 0)
track_normal_find_package(normal_len PNG)
track_rapids_find_package(rapids_len PNG)

if(NOT normal_len EQUAL rapids_len)
  message(FATAL_ERROR "A simple rapids_find_package leaked variables!")
endif()

track_normal_find_package(normal_len ZLIB)
track_rapids_find_package(rapids_len ZLIB
  INSTALL_EXPORT_SET test_export_set
  BUILD_EXPORT_SET test_export_set
  GLOBAL_TARGETS ZLIB::ZLIB
  )

if(NOT normal_len EQUAL rapids_len)
  message(FATAL_ERROR "A complex rapids_find_package leaked variables!")
endif()
