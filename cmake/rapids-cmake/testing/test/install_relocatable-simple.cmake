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

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cu"
     [=[
int main(int, char **)
{
  return 0;
}
]=])
add_executable(test_verify ${CMAKE_CURRENT_BINARY_DIR}/main.cu)

rapids_test_add(NAME verify_ COMMAND test_verify GPUS 1 INSTALL_COMPONENT_SET testing)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/testing)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake"
     "set(install_rules_file \"${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake\")")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake"
     [=[

file(READ "${install_rules_file}" contents)
set(exclude_from_all_string [===[if(CMAKE_INSTALL_COMPONENT STREQUAL "testing")]===])
string(FIND "${contents}" ${exclude_from_all_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "`rapids_test_install_relocatable` failed to mark items as EXCLUDE_FROM_ALL")
endif()
]=])
add_custom_target(verify_install_rule ALL
                  COMMAND ${CMAKE_COMMAND} -P
                          "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     "set(installed_test_file \"${CMAKE_CURRENT_BINARY_DIR}/install/bin/testing/CTestTestfile.cmake\")"
)
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
     [==[

file(READ "${installed_test_file}" contents)
set(add_test_match_string [===[add_test(generate_resource_spec ./generate_ctest_json "./resource_spec.json")]===])
string(FIND "${contents}" ${add_test_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate an installed `add_test` for generate_resource_spec")
endif()
set(add_test_match_string [===[add_test([=[verify_]=] "cmake" -Dcommand_to_run=${CMAKE_INSTALL_PREFIX}/bin/testing/test_verify -Dcommand_args= -P=./run_gpu_test.cmake)]===])
string(FIND "${contents}" ${add_test_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate an installed `add_test` for verify_")
endif()
set(properties_match_string [===[PROPERTIES RESOURCE_GROUPS 1,gpus:100]===])
string(FIND "${contents}" ${properties_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "${contents}\nFailed to generate an installed `GPU` requirements for verify_")
endif()
]==])

add_custom_target(install_testing_component ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component
                          testing --prefix install/ --config Debug
                  COMMAND ${CMAKE_COMMAND} -P
                          "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake")
add_dependencies(install_testing_component test_verify generate_ctest_json)
