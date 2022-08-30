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
include(${rapids-cmake-dir}/cmake/parse_version.cmake)

rapids_cmake_parse_version(PATCH "1.a.0" patch_value)
message(STATUS "patch_value: ${patch_value}")
if(NOT patch_value EQUAL 0)
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "0.200.git-sha1" patch_value)
if(NOT patch_value STREQUAL "git-sha1")
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()

rapids_cmake_parse_version(PATCH "21.03.00...22.04.00" patch_value)
if(NOT patch_value STREQUAL "00")
  message(FATAL_ERROR "rapids_cmake_parse_version(PATCH) value parsing failed unexpectedly")
endif()
