# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gbench.cmake)

rapids_cpm_init()

if(TARGET benchmark::benchmark)
  message(FATAL_ERROR "Expected benchmark::benchmark not to exist")
endif()

rapids_cpm_gbench()

if(NOT TARGET benchmark::benchmark)
  message(FATAL_ERROR "Expected benchmark::benchmark target to exist")
endif()

# Make sure we can be called multiple times
rapids_cpm_gbench()
