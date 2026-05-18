# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CONFIGURATION_TYPES Debug)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

set(CMAKE_INSTALL_MESSAGE LAZY)
enable_testing()
rapids_test_init()

add_executable(verify_ main.cu)
rapids_test_add(NAME verify
                COMMAND verify_
                GPUS 1
                PERCENT 100
                INSTALL_COMPONENT_SET testing)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/to_verify/)

add_custom_target(install_testing_component ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component
                          testing --prefix install/
                  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir install/bin/tests/to_verify/
                          --output-on-failure --no-tests=error)
add_dependencies(install_testing_component verify_ generate_ctest_json)
