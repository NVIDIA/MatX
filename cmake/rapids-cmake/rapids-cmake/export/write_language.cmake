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
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_export_write_language
----------------------------

.. versionadded:: v21.06.00

Creates a self-containted file that makes sure the requested language is enabled
globally.

.. code-block:: cmake

  rapids_export_write_language( (BUILD|INSTALL) (CXX|CUDA|...) <file_path> )


The contents of `<file_path>` will be a self-contained file that when called
via :cmake:command:`include <cmake:command:include>` will make sure the requested
language is enabled globally.

This is required as CMake's :cmake:command:`enable_language <cmake:command:enable_language>`
only supports enabling languages for the current directory scope, and doesn't support
being called from within functions. These limitations make it impossible
for packages included via `CPM` to enable extra languages.

.. note::
  This uses some serious CMake black magic to make sure that
  :cmake:command:`enable_language <cmake:command:enable_language>` occurs both at the call site,
  and up the entire :cmake:command:`enable_language <cmake:command:add_subdirectory>` stack so
  the language is enabled globally.


#]=======================================================================]
function(rapids_export_write_language type lang file_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.write_language")

  set(code_to_inject
      [=[
# Enable the requested language, which is only supported
# in the highest directory that 'uses' a language.
# We have to presume all directories use a language
# since linking to a target with language standards
# means `using`

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  if(NOT DEFINED CMAKE_CURRENT_FUNCTION)
    #Can't be called inside a function
    enable_language(@lang@)
    return()
  endif()
endif()

# If we aren't in the highest directory we need to hoist up
# all the language information to trick CMake into thinking
# the correct things happened.
# `cmake_langauge(DEFER )` doesn't support calling `enable_language`
# so we have to emulate what it does.
#
# So what we need to do is the following:
#
# 1. Transform each `set` in CMake@lang@Compiler to be a `PARENT_SCOPE`
#    This allows us to propagate up immediate information that is
#    used by commands such target_compile_features.
#
# 2. Make sure that every directory including root also re-executes
#    `CMake@lang@Information` This can't be defered as the contents
#    are required if any target is constructed
#

# Expose the language at the current scope
enable_language(@lang@)


if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/PropagateCMake@lang@Compiler.cmake")
  # 1.
  # Take everything that `enable_language` generated and transform all sets to PARENT_SCOPE ()
  # This will allow our parent directory to be able to call CMake@lang@Information
  file(STRINGS "${CMAKE_BINARY_DIR}/CMakeFiles/${CMAKE_VERSION}/CMake@lang@Compiler.cmake" rapids_code_to_transform)
  set(rapids_code_to_execute )
  foreach( line IN LISTS rapids_code_to_transform)
    if(line MATCHES "[ ]*set")
      string(REPLACE ")" " PARENT_SCOPE)" line "${line}")
    endif()
    string(APPEND rapids_code_to_execute "${line}\n")
  endforeach()

  # 2.
  # Make sure we call "CMake@lang@Information" for the current directory
  string(APPEND rapids_code_to_execute "include(CMake@lang@Information)\n")

  file(WRITE "${CMAKE_BINARY_DIR}/cmake/PropagateCMake@lang@Compiler.cmake" "${rapids_code_to_execute}")
  unset(rapids_code_to_execute)
  unset(rapids_code_to_transform)
endif()

# propagate up one parent_scope
include("${CMAKE_BINARY_DIR}/cmake/PropagateCMake@lang@Compiler.cmake")

# Compute all directories between here and the root of the project
# - Each directory but the root needs to include `PropagateCMake@lang@Compiler.cmake`
# - Since the root directory doesn't have a parent it only needs to include
#   `CMake@lang@Information`

set(rapids_directory "${CMAKE_CURRENT_SOURCE_DIR}")
if(DEFINED CMAKE_CURRENT_FUNCTION)
  string(APPEND rapids_directory "/fake_dir")
endif()

set(rapids_root_directory "${CMAKE_SOURCE_DIR}")
cmake_path(GET rapids_directory PARENT_PATH rapids_directory)
while(NOT rapids_directory STREQUAL rapids_root_directory)

  # Make sure we haven't already installed a language hook for this directory
  # Once we found a directory with an existing hook we can safely stop
  # as that means hooks exist from that point up in the graph
  cmake_language(DEFER DIRECTORY "${rapids_directory}" GET_CALL_IDS rapids_existing_calls)
  if(NOT rapids_@lang@_hook IN_LIST rapids_existing_calls)
    cmake_language(DEFER DIRECTORY "${rapids_directory}"
                   ID rapids_@lang@_hook
                   CALL include "${CMAKE_BINARY_DIR}/cmake/PropagateCMake@lang@Compiler.cmake")
  else()
    break()
  endif()

  cmake_path(GET rapids_directory PARENT_PATH rapids_directory)
endwhile()

# Make sure we haven't already installed a language hook for this directory
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" GET_CALL_IDS rapids_existing_calls)
if(NOT rapids_@lang@_hook IN_LIST rapids_existing_calls)
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
                 ID rapids_@lang@_hook
                 CALL include "CMake@lang@Information")
endif()

unset(rapids_existing_calls)
unset(rapids_directory)
unset(rapids_root_directory)
]=])

  string(CONFIGURE "${code_to_inject}" code_to_inject @ONLY)
  file(WRITE "${file_path}" "${code_to_inject}")

endfunction()
