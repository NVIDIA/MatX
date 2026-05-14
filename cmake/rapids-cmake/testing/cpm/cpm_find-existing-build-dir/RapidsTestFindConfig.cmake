# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/RapidsTestFindConfigVersion.cmake")

add_library(RapidsTest::RapidsTest IMPORTED INTERFACE GLOBAL)

check_required_components(RapidsTest2)
