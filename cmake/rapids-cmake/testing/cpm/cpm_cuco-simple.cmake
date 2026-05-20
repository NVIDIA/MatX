# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cuco.cmake)

rapids_cpm_init()

if(TARGET cuco::cuco)
  message(FATAL_ERROR "Expected cuco::cuco target not to exist")
endif()

rapids_cpm_cuco()
if(NOT TARGET cuco::cuco)
  message(FATAL_ERROR "Expected cuco::cuco target to exist")
endif()

# Ensure that calls are idempotent.
rapids_cpm_cuco()
