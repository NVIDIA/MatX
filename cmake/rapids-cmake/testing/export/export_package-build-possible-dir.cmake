#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

# Verify valid dir is picked up
set(FAKE_PACKAGE_DIR "/valid/looking/path")
rapids_export_package( build FAKE_PACKAGE test_export_set)

# Verify that package configuration files exist
set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_FAKE_PACKAGE.cmake")
if(NOT EXISTS "${path}")
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()

# verify that the expected path exists in FAKE_PACKAGE.cmake
set(to_match_string [=[set(possible_package_dir "/valid/looking/path")]=])
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()

# Verify in-valid dir is ignored
set(also_fake_package_DIR OFF)
rapids_export_package( BUILD also_fake_package test_export_set)
set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_also_fake_package.cmake")
if(NOT EXISTS "${path}")
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()

# verify that the expected path exists in also_fake_package.cmake
set(to_match_string [=[set(possible_package_dir "")]=])
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_package(BUILD) failed to write out the possible_package_dir")
endif()
