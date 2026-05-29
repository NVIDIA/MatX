# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/testing)

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
set(add_test_match_string [===[add_test([=[main]=] "cmake" -Dcommand_to_run=${CMAKE_INSTALL_PREFIX}/bin/testing/main -Dcommand_args= -P=./run_gpu_test.cmake)]===])
string(FIND "${contents}" ${add_test_match_string} is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to generate an installed `add_test` for main")
endif()
]==])

add_custom_target(install_testing_component ALL
                  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component
                          testing --prefix install/ --config Debug
                  COMMAND ${CMAKE_COMMAND} -P
                          "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake")
add_dependencies(install_testing_component main)
