#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

include(${rapids-cmake-dir}/rapids-test.cmake)

enable_testing()
rapids_test_init()

rapids_test_add(NAME verify_ COMMAND ls GPUS 1 INSTALL_COMPONENT_SET testing)
set_tests_properties(
    verify_
    PROPERTIES
      ENVIRONMENT
      "STREAM_MODE=new_mode;LD_PRELOAD=$<TARGET_FILE:preload>"
  )

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION bin/testing
                                INCLUDE_IN_ALL)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
  "set(installed_test_file \"${CMAKE_CURRENT_BINARY_DIR}/install/bin/testing/CTestTestfile.cmake\")")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
[==[

file(READ "${installed_test_file}" contents)
set(execute_process_match_string [===[execute_process(COMMAND ./generate_ctest_json OUTPUT_FILE "${CTEST_RESOURCE_SPEC_FILE}" COMMAND_ERROR_IS_FATAL ANY)]===])
string(FIND "${contents}" ${execute_process_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate a `execute_process` with escaped CTEST_RESOURCE_SPEC_FILE")
endif()

set(properties_match_strings [===[PROPERTIES ENVIRONMENT "STREAM_MODE]===]
                             [===[LD_PRELOAD=${CMAKE_INSTALL_PREFIX}/lib/libpreload]===])
foreach(to_match IN LISTS properties_match_strings)
  string(FIND "${contents}" ${to_match} is_found)
  if(is_found EQUAL -1)
    message(FATAL_ERROR "${contents}\nFailed to generate a proper set of test properties")
  endif()
endforeach()
]==])

add_custom_target(install_testing_component ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix install/ --config Debug
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
  )

add_dependencies(install_testing_component preload)
