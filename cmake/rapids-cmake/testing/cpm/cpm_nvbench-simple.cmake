# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)

rapids_cpm_init()

if(TARGET nvbench::nvbench)
  message(FATAL_ERROR "Expected nvbench::nvbench not to exist")
endif()

rapids_cpm_nvbench()

set(targets_made nvbench::nvbench nvbench::main)

foreach(t IN LISTS targets_made)
  if(NOT TARGET ${t})
    message(FATAL_ERROR "Expected ${t} target to exist")
  endif()
endforeach()

# Make sure we can be called multiple times
rapids_cpm_nvbench()
