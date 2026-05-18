# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# The only thing we can test is that everything comes back appended with -real
foreach(value IN LISTS CMAKE_CUDA_ARCHITECTURES)

  # verify it ends with `-real`
  string(FIND ${value} "-real" location)
  if(location LESS "0")
    message(FATAL_ERROR "All values in CMAKE_CUDA_ARCHITECTURES should have `-real`")
  endif()

  # verify that we don't append multiple `-real`
  string(FIND ${value} "real-real" location)
  if(NOT location EQUAL "-1")
    message(FATAL_ERROR "No values in CMAKE_CUDA_ARCHITECTURES should have `real-real`")
  endif()

endforeach()

if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
  message(FATAL_ERROR "rapids_cuda_set_architectures didn't make CMAKE_CUDA_ARCHITECTURES a cache variable"
  )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0 AND CMAKE_CUDA_COMPILER_VERSION
                                                                VERSION_LESS 13.0.0)
  if(NOT CMAKE_CUDA_FLAGS MATCHES "Wno-deprecated-gpu-targets")
    message(FATAL_ERROR "CMAKE_CUDA_FLAGS should have -Wno-deprecated-gpu-targets")
  endif()
endif()
