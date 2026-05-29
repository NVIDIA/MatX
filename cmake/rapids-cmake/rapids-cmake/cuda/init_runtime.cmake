# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cuda_init_runtime
-------------------------------

.. versionadded:: v21.06.00

Establish what CUDA runtime library should be propagated

  .. code-block:: cmake

    rapids_cuda_init_runtime( USE_STATIC (TRUE|FALSE) )

  Establishes what CUDA runtime will be used, if not already explicitly
  specified, via the :cmake:variable:`CMAKE_CUDA_RUNTIME_LIBRARY <cmake:variable:CMAKE_CUDA_RUNTIME_LIBRARY>`
  variable. We also set :cmake:variable:`CUDA_USE_STATIC_CUDA_RUNTIME <cmake:module:FindCUDA>` to control
  targets using the legacy `FindCUDA.cmake`

  When `USE_STATIC TRUE` is provided all targets will link to a
    statically-linked CUDA runtime library.

  When `USE_STATIC FALSE` is provided all targets will link to a
    shared-linked CUDA runtime library.


#]=======================================================================]
function(rapids_cuda_init_runtime use_static value)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.init_runtime")

  if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    if(${value})
      set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC PARENT_SCOPE)
    else()
      set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED PARENT_SCOPE)
    endif()
  endif()

  # Control legacy FindCUDA.cmake behavior too
  if(NOT DEFINED CUDA_USE_STATIC_CUDA_RUNTIME)
    if(${value})
      set(CUDA_USE_STATIC_CUDA_RUNTIME ON PARENT_SCOPE)
    else()
      set(CUDA_USE_STATIC_CUDA_RUNTIME OFF PARENT_SCOPE)
    endif()
  endif()

endfunction()
