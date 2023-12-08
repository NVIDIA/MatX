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
#
# This is the legacy entry point for projects using rapids-cmake.
#
# This will setup the following variables in the parent directory
# - CMAKE_MODULE_PATH
# - rapids-cmake-dir
#
# This is considered legacy as it has issues when multiple projects
# use rapids-cmake via CPM inside the same global project. In those
# cases it can fail due to CMAKE_MODULE_PREFIX not being exported properly

# Enforce the minimum required CMake version for all users
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

set(rapids-cmake-dir "${CMAKE_CURRENT_LIST_DIR}/rapids-cmake")
if(NOT DEFINED CACHE{rapids-cmake-dir})
  set(rapids-cmake-dir "${rapids-cmake-dir}" CACHE INTERNAL "" FORCE)
endif()

if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
endif()

# Propagate up the rapids-cmake version
include("${rapids-cmake-dir}/rapids-version.cmake")
set(rapids-cmake-version ${rapids-cmake-version} PARENT_SCOPE)

# install a hook that sets up `rapids-cmake-dir` and `CMAKE_MODULE_PATH` all the way up the
# call-stack
cmake_language(DEFER DIRECTORY ${CMAKE_CURRENT_LIST_DIR} CALL include
               "${rapids-cmake-dir}/../init.cmake")
