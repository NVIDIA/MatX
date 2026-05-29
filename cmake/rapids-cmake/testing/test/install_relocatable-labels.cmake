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

rapids_test_add(NAME verify_labels COMMAND ls GPUS 1 INSTALL_COMPONENT_SET testing)
set_tests_properties(verify_labels PROPERTIES LABELS "has_label")

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/testing)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     "set(installed_test_file \"${CMAKE_CURRENT_BINARY_DIR}/install/bin/testing/CTestTestfile.cmake\")"
)
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     [==[

file(READ "${installed_test_file}" contents)
set(labels_match_string [===[PROPERTIES LABELS has_label]===])
string(FIND "${contents}" "${labels_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to record the LABELS property")
endif()
]==])

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
