# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/cython-core/init.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cython-core/create_modules.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cython-core/add_rpath_entries.cmake)
