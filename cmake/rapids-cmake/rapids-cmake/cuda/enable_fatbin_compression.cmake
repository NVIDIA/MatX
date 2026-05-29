# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cuda_enable_fatbin_compression
-------------------------------------

.. versionadded:: v25.10.00

Set the needed compilation flags for FATBIN compression on a provided target and/or in
a variable

.. code-block:: cmake

  rapids_cuda_enable_fatbin_compression( [TARGET <target>]
                                         [VARIABLE <variable>]
                                         [TUNE_FOR (balanced|size|rapids|runtime_perf)] )


Establishes the needed compilation flags for the requested compression level,
by adding the required compilation flags for any CUDA sources in the target.

It is recommended for this function to be called without any `TUNE_FOR` option as
the value selection that has been tuned by RAPIDS.

``TARGET``
  If `target` doesn't currently exist it will be created as a global interface target.

``VARIABLE``
  A variable with the name specified in this option will be created with compilation
  flags needed to fulfull the compression level requested. If a variable already
  exists in the calling scope with this name it will be overwritten.

``balanced``
  Specify compression flags that aim for good tradeoff of binary size and
  runtime decompression performance.

  CUDA 12 Requires Driver Version: 550.54.14+
  CUDA 13 Requires Driver Version: 580+

``rapids``
  Specify compression flags that have been selected by RAPIDS. RAPIDS generally
  tunes towards minimal size but this is subject to change as the needs of
  RAPIDS evolves.

  CUDA 12 Requires Driver Version: 525.60.13+
  CUDA 13 Requires Driver Version: 580+

``runtime_perf``
  Specify compression flags that will aim for fastest runtime at the cost
  of compile time, and larger binary sizes.

  CUDA 12 Requires Driver Version: 525.60.13+
  CUDA 13 Requires Driver Version: 580+

``size``
  Specify compression flags that will aim for smallest binary size at the cost
  of compile time, and slower runtime decompression.

  CUDA 12 Requires Driver Version: 550.54.14+
  CUDA 13 Requires Driver Version: 580+


#]=======================================================================]
function(rapids_cuda_enable_fatbin_compression)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.enable_fatbin_compression")

  if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(VERBOSE
            "rapids_cuda_enable_fatbin_compression call ignored due to CUDA language not being enabled"
    )
    return()
  endif()

  set(options "")
  set(one_value TARGET TUNE_FOR VARIABLE)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _RAPIDS_TARGET AND NOT _RAPIDS_VARIABLE)
    message(FATAL_ERROR "rapids_cuda_enable_fatbin_compression requires either the `TARGET` or `VARIABLE` option be provided"
    )
  endif()

  # Common logic
  set(_rapids_cuda_flags "-Xfatbin=-compress-all")

  # CUDA 12.X logic
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 AND CMAKE_CUDA_COMPILER_VERSION
                                                                  VERSION_LESS 13.0.0)
    set(balanced_flag --compress-mode=balance)
    set(rapids_flag)
    set(runtime_perf_flag --compress-mode=speed)
    set(size_flag --compress-mode=size)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
      set(rapids_flag -Xfatbin=--compress-level=3)
    endif()
  endif()

  # CUDA 13.X logic
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0 AND CMAKE_CUDA_COMPILER_VERSION
                                                                  VERSION_LESS 14.0.0)
    set(balanced_flag --compress-mode=balance)
    set(rapids_flag --compress-mode=size)
    set(runtime_perf_flag --compress-mode=speed)
    set(size_flag --compress-mode=size)
  endif()

  string(TOLOWER "${_RAPIDS_TUNE_FOR}" _rapids_tune_for_lower)
  if(NOT _RAPIDS_TUNE_FOR)
    list(APPEND _rapids_cuda_flags "${rapids_flag}")
  elseif(${_rapids_tune_for_lower}_flag OR _rapids_tune_for_lower STREQUAL rapids)
    # rapids flag variable can be empty
    list(APPEND _rapids_cuda_flags "${${_rapids_tune_for_lower}_flag}")
  else()
    message(FATAL_ERROR "rapids_cuda_enable_fatbin_compression `TUNE_FOR` option was provided an unsupported value of ${_RAPIDS_TUNE_FOR}"
    )
  endif()

  if(_RAPIDS_VARIABLE)
    list(APPEND ${_RAPIDS_VARIABLE} ${_rapids_cuda_flags})
    set(${_RAPIDS_VARIABLE} "${${_RAPIDS_VARIABLE}}" PARENT_SCOPE)
  endif()

  if(_RAPIDS_TARGET)
    set(usage_requirement "PRIVATE")
    if(NOT TARGET ${_RAPIDS_TARGET})
      add_library(${_RAPIDS_TARGET} INTERFACE)
      set(usage_requirement "INTERFACE")
    endif()
    target_compile_options(${_RAPIDS_TARGET} ${usage_requirement}
                           "$<$<COMPILE_LANGUAGE:CUDA>:${_rapids_cuda_flags}>")
  endif()

endfunction()
