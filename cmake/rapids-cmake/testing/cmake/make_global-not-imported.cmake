# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/make_global.cmake)

add_library(real_target INTERFACE)
set(targets real_target)
rapids_cmake_make_global(targets)

get_target_property(is_global real_target IMPORTED_GLOBAL)
if(is_global)
  message(FATAL_ERROR "Expected is_global to be False [${is_global}]")
endif()
