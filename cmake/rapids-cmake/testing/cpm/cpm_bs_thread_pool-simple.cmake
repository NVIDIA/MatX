# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

rapids_cpm_init()

if(TARGET rapids_bs_thread_pool)
  message(FATAL_ERROR "Expected rapids_bs_thread_pool not to exist")
endif()

rapids_cpm_bs_thread_pool()

if(NOT TARGET rapids_bs_thread_pool)
  message(FATAL_ERROR "Expected rapids_bs_thread_pool target to exist")
endif()
if(NOT TARGET BS::thread_pool)
  message(FATAL_ERROR "Expected BS::thread_pool target to exist")
endif()

rapids_cpm_bs_thread_pool()

include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
rapids_cpm_generate_pinned_versions(OUTPUT ${CMAKE_BINARY_DIR}/versions.json)
