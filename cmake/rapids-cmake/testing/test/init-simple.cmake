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

enable_language(CUDA)

rapids_test_init()

if(NOT DEFINED CTEST_RESOURCE_SPEC_FILE)
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE should be set after calling rapids_test_init")
endif()

if(NOT EXISTS "${CTEST_RESOURCE_SPEC_FILE}")
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE should exist on disk after calling rapids_test_init")
endif()
