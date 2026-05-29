# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/make_global.cmake)

add_library(test_lib UNKNOWN IMPORTED)

set(targets test_lib)
rapids_cmake_make_global(targets)

get_target_property(is_global test_lib IMPORTED_GLOBAL)
if(NOT is_global)
  message(FATAL_ERROR "Expected is_global to be True [${is_global}]")
endif()
