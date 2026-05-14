# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/init.cmake)

enable_language(CUDA)

rapids_test_init()

if(NOT TEST generate_resource_spec)
  message(FATAL_ERROR "generate_resource_spec test should exist after calling rapids_test_init")
endif()

get_test_property(generate_resource_spec FIXTURES_SETUP fixtures_setup)
if(NOT fixtures_setup STREQUAL "resource_spec")
  message(FATAL_ERROR "generate_resource_spec FIXTURES_SETUP property should be set to \"resource_spec\""
  )
endif()

get_test_property(generate_resource_spec GENERATED_RESOURCE_SPEC_FILE grsf)
if(NOT grsf STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/resource_spec.json")
  message(FATAL_ERROR "generate_resource_spec FIXTURES_SETUP property should be set to \"${CMAKE_CURRENT_BINARY_DIR}/resource_spec.json\""
  )
endif()
