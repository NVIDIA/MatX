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
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CONFIGURATION_TYPES Debug)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

set(CMAKE_INSTALL_MESSAGE LAZY)
enable_testing()
rapids_test_init()

add_executable(verify_ main.cu)
rapids_test_add(
  NAME verify
  COMMAND verify_
  GPUS 1
  PERCENT 100
  INSTALL_COMPONENT_SET testing
)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/to_verify/)

add_custom_target(install_testing_component ALL
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}" --component testing --prefix install/
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir install/bin/tests/to_verify/ --output-on-failure --no-tests=error
  )
add_dependencies(install_testing_component verify_)
