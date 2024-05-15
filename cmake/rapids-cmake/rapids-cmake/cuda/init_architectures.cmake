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
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cuda_init_architectures
-------------------------------

.. versionadded:: v21.06.00

Extends :cmake:variable:`CMAKE_CUDA_ARCHITECTURES <cmake:variable:CMAKE_CUDA_ARCHITECTURES>`
to include support for `RAPIDS` and `NATIVE` to make CUDA architecture compilation easier.

  .. code-block:: cmake

    rapids_cuda_init_architectures(<project_name>)

Used before enabling the CUDA language either via :cmake:command:`project() <cmake:command:project>` to establish the
CUDA architectures to be compiled for. Parses the :cmake:envvar:`ENV{CUDAARCHS} <cmake:envvar:CUDAARCHS>`, and
:cmake:variable:`CMAKE_CUDA_ARCHITECTURES <cmake:variable:CMAKE_CUDA_ARCHITECTURES>` for special values
`RAPIDS`, and `NATIVE`.

.. note::
  Required to be called before the first :cmake:command:`project() <cmake:command:project>` call.

  Will automatically call :cmake:command:`rapids_cuda_set_architectures` immediately
  after :cmake:command:`project() <cmake:command:project>` with the same project name establishing
  the correct values for :cmake:variable:`CMAKE_CUDA_ARCHITECTURES <cmake:variable:CMAKE_CUDA_ARCHITECTURES>`.

``project_name``
  Name of the project in the subsequent :cmake:command:`project() <cmake:command:project>` call.

.. include:: supported_cuda_architectures_values.txt

Example on how to properly use :cmake:command:`rapids_cuda_init_architectures`:

.. code-block:: cmake

  cmake_minimum_required(...)

  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-<VERSION_MAJOR>.<VERSION_MINOR>/RAPIDS.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
  endif()
  include(${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
  include(rapids-cuda)

  rapids_cuda_init_architectures(ExampleProject)
  project(ExampleProject ...)




#]=======================================================================]
# cmake-lint: disable=W0105
function(rapids_cuda_init_architectures project_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.init_architectures")

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/architectures_policy.cmake)
  # If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
  # `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
  # architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.
  if(DEFINED ENV{CUDAARCHS} AND ("$ENV{CUDAARCHS}" STREQUAL "RAPIDS" OR "$ENV{CUDAARCHS}" STREQUAL
                                                                        "ALL"))
    set(cuda_arch_mode "$ENV{CUDAARCHS}")
    rapids_cuda_architectures_policy(FROM_INIT cuda_arch_mode)
  elseif(DEFINED ENV{CUDAARCHS} AND "$ENV{CUDAARCHS}" STREQUAL "NATIVE")
    set(cuda_arch_mode "NATIVE")
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "RAPIDS" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
    set(cuda_arch_mode "${CMAKE_CUDA_ARCHITECTURES}")
    rapids_cuda_architectures_policy(FROM_INIT cuda_arch_mode)
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(cuda_arch_mode "NATIVE")
    set(deprecated_cuda_arch_mode "EMPTY_STR")
    rapids_cuda_architectures_policy(FROM_INIT deprecated_cuda_arch_mode)
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
    set(cuda_arch_mode "NATIVE")
  elseif(NOT (DEFINED ENV{CUDAARCHS} OR DEFINED CMAKE_CUDA_ARCHITECTURES))
    set(cuda_arch_mode "RAPIDS")
  endif()

  # This needs to be run before enabling the CUDA language since RAPIDS supports magic values like
  # `RAPIDS`, `ALL`, and `NATIVE` which if propagated cause CMake to fail to determine the CUDA
  # compiler
  if(cuda_arch_mode STREQUAL "RAPIDS")
    set(CMAKE_CUDA_ARCHITECTURES OFF PARENT_SCOPE)
    set(load_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/invoke_set_all_architectures.cmake")
  elseif(cuda_arch_mode STREQUAL "NATIVE")
    set(CMAKE_CUDA_ARCHITECTURES OFF PARENT_SCOPE)
    set(load_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/invoke_set_native_architectures.cmake")
  endif()

  if(load_file)
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/set_architectures.cmake")

    # Setup to call to set CMAKE_CUDA_ARCHITECTURES values to occur right after the project call
    # https://cmake.org/cmake/help/latest/command/project.html#code-injection
    #
    # If an existing file was specified for loading post `project` we will chain include them
    if(DEFINED CMAKE_PROJECT_${project_name}_INCLUDE)
      set(_RAPIDS_PREVIOUS_CMAKE_PROJECT_INCLUDE "${CMAKE_PROJECT_${project_name}_INCLUDE}"
          PARENT_SCOPE)
    endif()
    set(CMAKE_PROJECT_${project_name}_INCLUDE "${load_file}" PARENT_SCOPE)
  endif()

endfunction()
