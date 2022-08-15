#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)


rapids_export_cpm(BUILD RaFT fake_set
                  CPM_ARGS
                    FAKE_PACKAGE_ARGS TRUE
                  )

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export(BUILD FakEProJecT
  EXPORT_SET fake_set
  LANGUAGES CXX
  )

# Verify that build files have correct names
if(NOT EXISTS "${CMAKE_BINARY_DIR}/fakeproject-config.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config file name")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/fakeproject-config-version.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct config-version file name")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/fakeproject-dependencies.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct dependencies file name")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/fakeproject-CXX-language.cmake")
  message(FATAL_ERROR "rapids_export failed to generate correct language file name")
endif()
