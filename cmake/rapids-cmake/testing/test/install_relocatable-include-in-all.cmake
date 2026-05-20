# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/add.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CUDA)
enable_testing()

rapids_test_init()
rapids_test_add(NAME verify_ COMMAND ls GPUS 1 INSTALL_COMPONENT_SET testing)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/testing
                                INCLUDE_IN_ALL)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install_rules.cmake"
     "set(install_rules_file \"${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake\")")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install_rules.cmake"
     [=[

file(READ "${install_rules_file}" contents)
set(exclude_from_all_string [===[if(CMAKE_INSTALL_COMPONENT STREQUAL "testing" OR NOT CMAKE_INSTALL_COMPONENT)]===])
string(FIND "${contents}" ${exclude_from_all_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "`rapids_test_install_relocatable` failed to mark items as EXCLUDE_FROM_ALL")
endif()
]=])

add_custom_target(verify_install_rule ALL
                  COMMAND ${CMAKE_COMMAND} -P
                          "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install_rules.cmake")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     "set(installed_test_file \"${CMAKE_CURRENT_BINARY_DIR}/install/bin/testing/CTestTestfile.cmake\")"
)
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     [=[

file(READ "${installed_test_file}" contents)
set(add_test_match_string [===[add_test(generate_resource_spec ./generate_ctest_json "./resource_spec.json")]===])
string(FIND "${contents}" ${add_test_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate a `execute_process` with escaped CTEST_RESOURCE_SPEC_FILE")
endif()
]=])

set(_config_arg)
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
  set(_config_arg --config $<CONFIG>)
endif()
add_custom_target(install_testing_component ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component
                          testing --prefix install/ ${_config_arg}
                  COMMAND ${CMAKE_COMMAND} -P
                          "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake")
add_dependencies(install_testing_component generate_ctest_json)
