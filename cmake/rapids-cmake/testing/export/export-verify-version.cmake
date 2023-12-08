#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/export.cmake)

project(test LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export(BUILD test
  EXPORT_SET fake_set
  LANGUAGES CXX
  )

# Verify that build files have correct names
if(NOT EXISTS "${CMAKE_BINARY_DIR}/test-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()

unset(TEST_VERSION)
unset(TEST_VERSION_MAJOR)
unset(TEST_VERSION_MINOR)

set(CMAKE_FIND_PACKAGE_NAME test) # Emulate `find_package`
include("${CMAKE_BINARY_DIR}/test-config.cmake")

if(NOT TEST_VERSION VERSION_EQUAL 3.1.4)
  message(FATAL_ERROR "rapids_export failed to export version information")
endif()

if(NOT "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}" VERSION_EQUAL 3.1)
  message(FATAL_ERROR "rapids_export failed to export version major/minor information")
endif()
