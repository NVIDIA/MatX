# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/generate_resource_spec.cmake)

enable_language(CUDA)

set(CTEST_RESOURCE_SPEC_FILE "sentinel file")
rapids_test_generate_resource_spec(DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/spec.json)

if(NOT CTEST_RESOURCE_SPEC_FILE STREQUAL "sentinel file")
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE shouldn't be modified by calling rapids_test_generate_resource_spec"
  )
endif()

if(NOT TEST generate_resource_spec)
  message(FATAL_ERROR "rapids_test_generate_resource_spec failed to create the generate_resource_spec test"
  )
endif()
