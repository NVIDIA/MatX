# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/cuda/init_architectures.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cuda/init_runtime.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cuda/set_architectures.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cuda/set_runtime.cmake)
