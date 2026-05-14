# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/make_global.cmake)

add_library(test_lib UNKNOWN IMPORTED)
add_library(test::lib ALIAS test_lib)

set(targets test::lib)
rapids_cmake_make_global(targets)

# verify that test::lib AND test_lib are now global
get_target_property(alias_is_global test::lib ALIAS_GLOBAL)
if(alias_is_global)
  message(FATAL_ERROR "Expected alias_is_global to be False [${alias_is_global}]")
endif()
