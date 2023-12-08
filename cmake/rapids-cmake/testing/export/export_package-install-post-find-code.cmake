#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/export/package.cmake)
include(${rapids-cmake-dir}/export/detail/post_find_package_code.cmake)

rapids_export_package(INSTALL FAKE_PACKAGE test_export_set VERSION 22.08)
rapids_export_post_find_package_code(INSTALL FAKE_PACKAGE "set(a ON)" EXPORT_SET  test_export_set)

get_target_property(install_code rapids_export_install_test_export_set FAKE_PACKAGE_POST_FIND_CODE)
cmake_language(EVAL CODE "${install_code}")
if(NOT a)
  message(FATAL_ERROR "rapids_export_post_find_package_code failed to record first call")
endif()

include(${rapids-cmake-dir}/export/write_dependencies.cmake)
rapids_export_write_dependencies(INSTALL test_export_set "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake")

set(to_match [=[if(FAKE_PACKAGE_FOUND)_set(a ON)__endif()]=])

file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/install_export_set.cmake" contents NEWLINE_CONSUME)
string(REPLACE "\n" "_" contents "${contents}")

string(FIND "${contents}" "${to_match}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(INSTALL) failed to record rapids_export_post_find_package_code")
endif()
