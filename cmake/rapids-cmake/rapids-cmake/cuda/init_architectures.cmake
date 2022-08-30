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
rapids_cuda_init_architectures
-------------------------------

.. versionadded:: v21.06.00

Extends :cmake:variable:`CMAKE_CUDA_ARCHITECTURES` to include support
for `ALL` and `NATIVE` to make CUDA architecture compilation easier.

  .. code-block:: cmake

    rapids_cuda_init_architectures(<project_name>)

Establishes what CUDA architectures that will be compiled for.
Parses the :cmake:variable:`CMAKE_CUDA_ARCHITECTURES` for special
values `ALL`, `NATIVE` and `""`.

.. note::
  Required to be called before :cmake:command:`project() <cmake:command:project>`

  Will automatically call :cmake:command:`rapids_cuda_set_architectures` immediately
  after :cmake:command:`project() <cmake:command:project>` establishing the correct values for
  :cmake:variable:`CMAKE_CUDA_ARCHITECTURES`.

``project_name``
  Name of the project

``NATIVE`` or ``""``:
  When passed will compile for all GPU architectures present on the current machine

``ALL``:
  When passed will compile for all supported RAPIDS GPU architectures


Example on how to properly use :cmake:command:`rapids_cuda_init_architectures`:

.. code-block:: cmake

  cmake_minimum_required(...)

  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-<VERSION_MAJOR>.<VERSION_MINOR>/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  include(rapids-cuda)

  rapids_cuda_init_architectures(ExampleProject)
  project(ExampleProject ...)




#]=======================================================================]
# cmake-lint: disable=W0105
function(rapids_cuda_init_architectures project_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.init_architectures")
  # If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
  # `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
  # architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

  # This needs to be run before enabling the CUDA language since RAPIDS supports the magic string of
  # "ALL"
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
    set(CMAKE_CUDA_ARCHITECTURES OFF PARENT_SCOPE)
    set(load_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/invoke_set_all_architectures.cmake")
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
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
