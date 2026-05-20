# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/init.cmake)

enable_language(CUDA)

set(CTEST_RESOURCE_SPEC_FILE "sentinel file")
rapids_test_init()

if(NOT CTEST_RESOURCE_SPEC_FILE STREQUAL "sentinel file")
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE shouldn't be modified if already set before calling rapids_test_init"
  )
endif()
