# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

include(utils/cmake_test.cmake)

function(add_cmake_build_test source_or_dir)
  add_cmake_test(BUILD "${source_or_dir}" ${ARGN})
endfunction()
