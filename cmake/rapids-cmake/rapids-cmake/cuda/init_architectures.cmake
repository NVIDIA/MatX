# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cuda_init_architectures
-------------------------------

.. versionadded:: v21.06.00

Extends :cmake:variable:`CMAKE_CUDA_ARCHITECTURES <cmake:variable:CMAKE_CUDA_ARCHITECTURES>`
to include support for `RAPIDS` and `NATIVE` to make CUDA architecture compilation easier.

  .. code-block:: cmake

    rapids_cuda_init_architectures(<project_name>)

Used before enabling the CUDA language either via :cmake:command:`project() <cmake:command:project>` or
:cmake:command:`enable_language() <cmake:command:enable_language>` to establish the CUDA architectures
to be compiled for. Parses the :cmake:envvar:`ENV{CUDAARCHS} <cmake:envvar:CUDAARCHS>`, and
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
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/main/RAPIDS.cmake
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

  # If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
  # `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
  # architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Don't ever look at `ENV{CUDAARCHS}` if a `CMAKE_CUDA_ARCHITECTURES` has been specified
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "RAPIDS" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
      set(cuda_arch_mode "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  elseif(DEFINED ENV{CUDAARCHS} AND ("$ENV{CUDAARCHS}" STREQUAL "RAPIDS" OR "$ENV{CUDAARCHS}"
                                                                            STREQUAL "NATIVE"))
    set(cuda_arch_mode "$ENV{CUDAARCHS}")
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
    list(APPEND CMAKE_PROJECT_${project_name}_INCLUDE "${load_file}")
    set(CMAKE_PROJECT_${project_name}_INCLUDE "${CMAKE_PROJECT_${project_name}_INCLUDE}"
        PARENT_SCOPE)
  endif()

endfunction()
