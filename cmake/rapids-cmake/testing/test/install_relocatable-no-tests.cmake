#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/test/add.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CUDA)
enable_testing()

rapids_test_add(NAME verify_ COMMAND ls GPUS 1 INSTALL_COMPONENT_SET testing)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION bin/testing)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake"
"set(install_rules_file \"${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake\")")

file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake"
[=[

file(READ "${install_rules_file}" contents)
set(bad_install_rule_match_string [===[rapids-cmake/generate_ctest_json")]===])
string(FIND "${contents}" ${bad_install_rule_match_string} is_found)
if(NOT is_found EQUAL -1)
  message(FATAL_ERROR "`rapids_test_install_relocatable` trying to install files that don't exist")
endif()
]=])
add_custom_target(verify_install_files_valid ALL
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/verify_cmake_install.cmake")
