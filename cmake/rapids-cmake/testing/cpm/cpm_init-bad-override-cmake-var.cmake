# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)

set(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE ${CMAKE_CURRENT_LIST_DIR}/bad_path.cmake)
rapids_cpm_init()
