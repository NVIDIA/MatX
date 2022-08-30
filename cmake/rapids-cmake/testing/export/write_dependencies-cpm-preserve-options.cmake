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
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_cpm( INSTALL RMM test_set
                   CPM_ARGS
                    NAME RMM
                    VERSION 2.0
                    OPTIONS
                      "FAKE_PACKAGE_ARGS FALSE"
                   GLOBAL_TARGETS RMM::RMM_POOL
                   )

rapids_export_write_dependencies(install test_set "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake")

# Parse the `export_set.cmake` file for correct escaped args to `CPMFindPackage` calls
set(to_match_string [=["NAME;RMM;VERSION;2.0;OPTIONS;FAKE_PACKAGE_ARGS FALSE"]=])

file(READ "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(BUILD) failed to perserve quotes around CPM arguments")
endif()
