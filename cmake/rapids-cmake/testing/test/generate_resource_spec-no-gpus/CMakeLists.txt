# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/test/generate_resource_spec.cmake)

enable_testing()

rapids_test_generate_resource_spec(DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/spec.json)

add_test(NAME read_spec_file COMMAND ${CMAKE_COMMAND} -P
                                     ${CMAKE_CURRENT_SOURCE_DIR}/read_spec_file.cmake)
set_tests_properties(read_spec_file PROPERTIES FIXTURES_REQUIRED resource_spec)
# Set the visible device list to invalid device ids so we have zero visible
set_tests_properties(generate_resource_spec PROPERTIES ENVIRONMENT CUDA_VISIBLE_DEVICES=-1)

set(_config_arg)
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
  set(_config_arg -C $<CONFIG>)
endif()
add_custom_target(run_tests ALL
                  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_CURRENT_BINARY_DIR}
                          --no-tests=error -V ${_config_arg})
add_dependencies(run_tests generate_ctest_json)
