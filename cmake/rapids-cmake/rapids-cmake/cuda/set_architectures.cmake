#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
rapids_cuda_set_architectures
-------------------------------

.. versionadded:: v21.06.00

Sets up :cmake:variable:`CMAKE_CUDA_ARCHITECTURES` based on the requested mode

.. code-block:: cmake

  rapids_cuda_set_architectures( (NATIVE|RAPIDS) )

Establishes what CUDA architectures that will be compiled for, overriding
any existing :cmake:variable:`CMAKE_CUDA_ARCHITECTURES` value.

This function should rarely be used, as :cmake:command:`rapids_cuda_init_architectures`
allows for the expected workflow of using :cmake:variable:`CMAKE_CUDA_ARCHITECTURES`
when configuring a project. If for some reason your project can't use
:cmake:command:`rapids_cuda_init_architectures` than you can use :cmake:command:`rapids_cuda_set_architectures`
directly.

.. note::
  This is automatically called by :cmake:command:`rapids_cuda_init_architectures`

.. include:: supported_cuda_architectures_values.txt

Result Variables
^^^^^^^^^^^^^^^^

``CMAKE_CUDA_ARCHITECTURES``

  Will exist as a local variable and be set to the list of architectures
  that should be compiled for. Will overwrite any existing values.

.. versionadded:: v24.08.00
  Will be added as a cache variable if it isn't already one.

#]=======================================================================]
function(rapids_cuda_set_architectures mode)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.set_architectures")

  set(supported_archs "70" "75" "80" "86" "90")

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1.0)
    list(REMOVE_ITEM supported_archs "86")
  endif()
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8.0)
    list(REMOVE_ITEM supported_archs "90")
  endif()

  if(${mode} STREQUAL "RAPIDS")

    # CMake architecture list entry of "80" means to build compute and sm. What we want is for the
    # newest arch only to build that way while the rest built only for sm.
    list(POP_BACK supported_archs latest_arch)
    list(TRANSFORM supported_archs APPEND "-real")
    list(APPEND supported_archs ${latest_arch})

    set(CMAKE_CUDA_ARCHITECTURES ${supported_archs})
  elseif(${mode} STREQUAL "NATIVE")
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/detect_architectures.cmake)
    rapids_cuda_detect_architectures(supported_archs CMAKE_CUDA_ARCHITECTURES)

    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
  endif()

  # cache the cuda archs.
  get_property(cached_value GLOBAL PROPERTY rapids_cuda_architectures)
  if(NOT cached_value)
    set_property(GLOBAL PROPERTY rapids_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  if(NOT cached_value STREQUAL CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE ";" "\n  " _cuda_architectures_pretty "${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Project ${PROJECT_NAME} is building for CUDA architectures:\n  ${_cuda_architectures_pretty}"
    )
  endif()

  # Need to set in the cache so we match CMake behavior of setting up as a cache variable. This
  # ensures that on subsequent executions we use the value we computed from the first execution, and
  # don't re-evaluate env vars which could have changed
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures")

  # Set as a local variable to maintain comp
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)

endfunction()
