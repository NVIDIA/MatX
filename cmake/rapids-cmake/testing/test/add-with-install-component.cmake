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

enable_language(CUDA)

rapids_test_init()

file(WRITE "${CMAKE_BINARY_DIR}/main.cu" "int main(){return 0;}")
add_executable(verify_alloc "${CMAKE_BINARY_DIR}/main.cu")

enable_testing()
rapids_test_add(NAME simple_test COMMAND verify_alloc GPUS 1 INSTALL_COMPONENT_SET testing)

# Verify that we have recorded `simple_test` as part of the `testing` component
get_target_property(names rapids_test_install_testing TESTS_TO_RUN)
if(NOT "simple_test" IN_LIST names)
  message(FATAL_ERROR "Failed to record `simple_test` as part of the testing component")
endif()

# Verify that `verify_alloc` is marked as to be installed
get_target_property(names rapids_test_install_testing TARGETS_TO_INSTALL)
if(NOT "verify_alloc" IN_LIST names)
  message(FATAL_ERROR "Failed to record `verify_alloc` as a target to be installed in the testing component")
endif()
