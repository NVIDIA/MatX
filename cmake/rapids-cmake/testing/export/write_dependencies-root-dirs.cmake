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
include(${rapids-cmake-dir}/export/find_package_root.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_find_package_root(BUILD RMM [=[${CMAKE_CURRENT_LIST_DIR}/fake/build/path]=] EXPORT_SET test_set)
rapids_export_write_dependencies(build test_set "${CMAKE_CURRENT_BINARY_DIR}/build_export_set.cmake")

rapids_export_find_package_root(BUILD RMM "/bad/install/path" EXPORT_SET ${unknown_var}) #ignored
rapids_export_find_package_root(BUILD RMM "/bad/install/path2" EXPORT_SET test_set CONDITION unknown_var) #ignored

# Parse the `build_export_set.cmake` file for correct escaped args
# to `rapids_export_find_package_root` calls
set(build_to_match_string [=[set(RMM_ROOT "${CMAKE_CURRENT_LIST_DIR}/fake/build/path"]=])
file(READ "${CMAKE_CURRENT_BINARY_DIR}/build_export_set.cmake" contents)
string(FIND "${contents}" "${build_to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(BUILD) failed to preserve variables in the directory path to rapids_export_find_package_root")
endif()

rapids_export_find_package_root(install RMM "/first/install/path" EXPORT_SET test_set)
rapids_export_find_package_root(INSTALL RMM "/second/install/path" EXPORT_SET test_set)
set(to_install FALSE)
rapids_export_find_package_root(INSTALL RMM "/bad/install/path" EXPORT_SET test_set CONDITION to_install) #ignored

rapids_export_find_package_root(install PKG2 "/pkg2/install/path" EXPORT_SET test_set)
rapids_export_write_dependencies(INSTALL test_set "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake")

set(install_to_match_string_1 [=[set(RMM_ROOT "/first/install/path;/second/install/path"]=])
set(install_to_match_string_2 [=[set(PKG2_ROOT "/pkg2/install/path"]=])
file(READ "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake" contents)
string(FIND "${contents}" "${install_to_match_string_1}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(INSTALL) failed to record all RMM_ROOT rapids_export_find_package_root commands")
endif()
string(FIND "${contents}" "${install_to_match_string_2}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(INSTALL) failed to record all PKG2 rapids_export_find_package_root commands")
endif()
