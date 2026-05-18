# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_subdirectory(D)

add_library(B STATIC static.not_cu)
set_source_files_properties(static.not_cu PROPERTIES LANGUAGE CUDA)
