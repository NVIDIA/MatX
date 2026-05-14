# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init()

# Test that CCCL::cudax does not exist before calling with ENABLE_UNSTABLE
if(TARGET CCCL::cudax)
  message(FATAL_ERROR "Expected CCCL::cudax not to exist")
endif()

# Call rapids_cpm_cccl with ENABLE_UNSTABLE
rapids_cpm_cccl(ENABLE_UNSTABLE)

# Test that all expected targets exist, including CCCL::cudax
set(targets CCCL::CCCL CCCL::CUB CCCL::libcudacxx CCCL::Thrust CCCL::cudax)
foreach(target IN LISTS targets)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Expected ${target} to exist")
  endif()
endforeach()

# Test that calling rapids_cpm_cccl(ENABLE_UNSTABLE) again is idempotent
rapids_cpm_cccl(ENABLE_UNSTABLE)

# Since rapids_cpm_cccl(ENABLE_UNSTABLE) is run already, subsequent calls to rapids_cpm_cccl()
# should also show CCCL::cudax in TARGETS. this is important when using the same EXPORT_SET for
# multiple calls to rapids_cpm_cccl(). Example: rapidsmpf requires CCCL::cudax, while also depending
# on rmm and cudf (which don't require CCCL::cudax). So, rapids_cpm_cccl() calls in rmm and cudf
# should not remove CCCL::cudax from TARGETS.
rapids_cpm_cccl()
foreach(target IN LISTS targets)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Expected ${target} to exist after third call")
  endif()
endforeach()
