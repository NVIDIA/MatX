# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(B)

add_library(A STATIC static.cu)
