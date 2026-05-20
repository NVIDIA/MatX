# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)

rapids_cpm_init()

if(TARGET rmm::rmm)
  message(FATAL_ERROR "Expected rmm::rmm not to exist")
endif()

rapids_cpm_rmm()
if(NOT TARGET rmm::rmm)
  message(FATAL_ERROR "Expected rmm::rmm target to exist")
endif()

rapids_cpm_rmm()
