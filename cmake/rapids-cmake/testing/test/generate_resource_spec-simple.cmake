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
include(${rapids-cmake-dir}/test/generate_resource_spec.cmake)

enable_language(CUDA)

set(CTEST_RESOURCE_SPEC_FILE "sentinel file")
rapids_test_generate_resource_spec(DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/spec.json )

if(NOT CTEST_RESOURCE_SPEC_FILE STREQUAL "sentinel file")
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE shouldn't be modified by calling rapids_test_generate_resource_spec")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/spec.json")
  message(FATAL_ERROR "rapids_test_generate_resource_spec failed to write out the requested spec file")
endif()
