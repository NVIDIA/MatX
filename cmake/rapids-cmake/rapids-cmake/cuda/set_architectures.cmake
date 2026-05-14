# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
      set(supported_archs "75-real" "80-real" "86-real" "90a-real" "100f-real" "120a-real" "120")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
      set(supported_archs "70-real" "75-real" "80-real" "86-real" "90a-real" "100f-real"
                          "120a-real" "120")
    else()
      set(supported_archs "70-real" "75-real" "80-real" "86-real" "90a-real" "90-virtual")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
        list(REMOVE_ITEM supported_archs "90-virtual")
        list(APPEND supported_archs "100-real" "120a-real" "120-virtual")
      endif()
    endif()

    # For the CUDA 12.X.0 series we want to silence warnings when compiling for arch 70 when
    # compiling for RAPIDS architectures.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0 AND CMAKE_CUDA_COMPILER_VERSION
                                                                    VERSION_LESS 13.0.0)
      string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
    endif()
  endif()

  if(${mode} STREQUAL "RAPIDS")
    set(CMAKE_CUDA_ARCHITECTURES ${supported_archs})
  elseif(${mode} STREQUAL "NATIVE")
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/detect_architectures.cmake)
    rapids_cuda_detect_architectures(supported_archs CMAKE_CUDA_ARCHITECTURES)
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
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} PARENT_SCOPE)

endfunction()
