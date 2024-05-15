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
cmake_minimum_required(VERSION 3.20)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

rapids_test_init()

add_executable(verify_alloc main.cu)

enable_testing()
rapids_test_add(NAME simple_test COMMAND verify_alloc GPUS 10 PERCENT 1)

# Validate that the test runs correctly
set_tests_properties(simple_test
  PROPERTIES PASS_REGULAR_EXPRESSION "Seeing at least a single GPU"
  )
