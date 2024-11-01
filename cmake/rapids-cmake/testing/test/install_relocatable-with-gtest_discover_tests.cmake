#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/add.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CXX)

rapids_cpm_init()
rapids_cpm_gtest()

enable_testing()
rapids_test_init()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp"
[=[
#include <gtest/gtest.h>

using Foo = ::testing::Test;

TEST_F(Foo, Bar) { }
TEST_F(Foo, Woo) { }

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
]=])
add_executable(main ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)

target_link_libraries(main PRIVATE GTest::gtest)


gtest_discover_tests(main DISCOVERY_MODE PRE_TEST)
rapids_test_add(NAME main COMMAND main INSTALL_COMPONENT_SET testing)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION bin/testing)


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
set(add_test_match_string [===[add_test([=[main]=] "cmake" -Dcommand_to_run=${CMAKE_INSTALL_PREFIX}/bin/testing/main -Dcommand_args= -P=./run_gpu_test.cmake)]===])
string(FIND "${contents}" ${add_test_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate an installed `add_test` for main")
endif()
]==])

add_custom_target(install_testing_component ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component testing --prefix install/ --config Debug
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
  )
add_dependencies(install_testing_component main)
