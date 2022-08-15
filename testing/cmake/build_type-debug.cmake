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
include(${rapids-cmake-dir}/cmake/build_type.cmake)

rapids_cmake_build_type(DEBUG)

if(CMAKE_CONFIGURATION_TYPES)
  if(DEFINED CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "rapids_cmake_build_type failed when executed by a multi-config generator")
  endif()
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  message(FATAL_ERROR "rapids_cmake_build_type failed")
endif()
