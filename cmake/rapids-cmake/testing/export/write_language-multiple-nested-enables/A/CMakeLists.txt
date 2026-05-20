# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(B)
add_subdirectory(C)

# Verify number of hooks is equal to 1
cmake_language(DEFER GET_CALL_IDS all_ids)

list(LENGTH all_ids list_len)
if(NOT list_len EQUAL 2)
  message(FATAL_ERROR "incorrect number of language hooks detected")
endif()

add_library(A STATIC static.cu)
