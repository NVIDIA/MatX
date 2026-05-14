# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init()

set(targets CCCL::CCCL CCCL::CUB CCCL::libcudacxx CCCL::Thrust)
foreach(target IN LISTS targets)
  if(TARGET ${target})
    message(FATAL_ERROR "Expected ${target} not to exist")
  endif()
endforeach()

rapids_cpm_cccl()

foreach(target IN LISTS targets)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Expected ${target} not to exist")
  endif()
endforeach()

rapids_cpm_cccl()
