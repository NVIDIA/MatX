#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/add.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CUDA)
enable_testing()
rapids_test_init()

rapids_test_add(NAME verify_labels COMMAND ls GPUS 1 INSTALL_COMPONENT_SET testing)
set_tests_properties(verify_labels PROPERTIES LABELS "has_label")

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION bin/testing)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
  "set(installed_test_file \"${CMAKE_CURRENT_BINARY_DIR}/install/bin/testing/CTestTestfile.cmake\")")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
[==[

file(READ "${installed_test_file}" contents)
set(labels_match_string [===[PROPERTIES LABELS has_label]===])
string(FIND "${contents}" "${labels_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "Failed to record the LABELS property")
endif()
]==])

add_custom_target(install_testing_component ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component testing --prefix install/
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/verify_installed_CTestTestfile.cmake"
  )
