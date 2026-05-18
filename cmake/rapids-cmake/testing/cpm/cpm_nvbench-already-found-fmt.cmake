# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)

# Fake bring in fmt via find package
add_library(fmt::fmt INTERFACE IMPORTED GLOBAL)

rapids_cpm_init()
rapids_cpm_nvbench()
if(fmt_ADDED)
  message(FATAL_ERROR "fmt shouldn't be added if it exists via `find_package`")
endif()
