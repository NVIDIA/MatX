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


rapids_export_cpm( build
                   FAKE_CPM_PACKAGE
                   test_export_set
                   CPM_ARGS
                      NAME FAKE_CPM_PACKAGE
                      VERSION 1.0
                      OPTIONS
                        "FAKE_PACKAGE_OPTION_A TRUE"
                        "FAKE_PACKAGE_OPTION_B FALSE"
                   )

rapids_export_cpm( install
                   FAKE_CPM_PACKAGE
                   test_export_set
                   CPM_ARGS
                      NAME FAKE_CPM_PACKAGE
                      VERSION 1.0
                      OPTIONS
                        "FAKE_PACKAGE_OPTION_A TRUE"
                        "FAKE_PACKAGE_OPTION_B FALSE"
                   )

if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake")
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to generate a CPM configuration")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/cpm_FAKE_CPM_PACKAGE.cmake")
  message(FATAL_ERROR "rapids_export_cpm(INSTALL) failed to generate a CPM configuration")
endif()

# We need to validate both of the files all CPM args in quotes
#
set(to_match_string [=["NAME;FAKE_CPM_PACKAGE;VERSION;1.0;OPTIONS;FAKE_PACKAGE_OPTION_A TRUE;FAKE_PACKAGE_OPTION_B FALSE"]=])

file(READ "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to perserve quotes around CPM arguments")
endif()

file(READ "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/cpm_FAKE_CPM_PACKAGE.cmake" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(INSTALL) failed to perserve quotes around CPM arguments")
endif()
