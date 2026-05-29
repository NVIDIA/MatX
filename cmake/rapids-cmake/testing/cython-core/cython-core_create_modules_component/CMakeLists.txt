# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.20)

include(${rapids-cmake-dir}/cython-core/create_modules.cmake)
include(${rapids-cmake-dir}/cython-core/init.cmake)

project(rapids_cython-create_modules LANGUAGES C CXX)

rapids_cython_init()

rapids_cython_create_modules(SOURCE_FILES test1.pyx COMPONENT Component1)
rapids_cython_create_modules(SOURCE_FILES test2.pyx COMPONENT Component2)

# Test that component install works correctly
set(_config_arg)
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
  set(_config_arg --config $<CONFIG>)
endif()
add_custom_target(check ALL
                  COMMAND ${CMAKE_COMMAND} --install . --component Component1 --prefix
                          install-component1/ ${_config_arg}
                  COMMAND ${CMAKE_COMMAND} --install . --component Component2 --prefix
                          install-component2/ ${_config_arg}
                  COMMAND ${CMAKE_COMMAND} -P
                          ${CMAKE_CURRENT_SOURCE_DIR}/check_installed_components.cmake)
add_dependencies(check test1 test2)
