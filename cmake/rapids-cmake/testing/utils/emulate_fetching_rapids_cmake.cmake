#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

# Emulate the variables and properties that FetchContent would set so
# tests that themselves download rapids-cmake will use the version we have
# symlinked.
include(FetchContent)

set(prefix "_FetchContent_rapids-cmake")
get_property(_rapids_already_hooked GLOBAL PROPERTY ${prefix}_populated DEFINED)
if(NOT _rapids_already_hooked)
  set(local-rapids-cmake-root "${rapids-cmake-dir}/..")
  cmake_path(NORMAL_PATH local-rapids-cmake-root)

  set(scratch_dir "${CMAKE_CURRENT_BINARY_DIR}/_deps")

  set(rapids-cmake_SOURCE_DIR "${scratch_dir}/rapids-cmake-src")
  set(rapids-cmake_BINARY_DIR "${scratch_dir}/rapids-cmake-build")
  set(rapids-cmake_POPULATED TRUE)
  set_property(GLOBAL PROPERTY ${prefix}_sourceDir "${rapids-cmake_SOURCE_DIR}")
  set_property(GLOBAL PROPERTY ${prefix}_binaryDir "${rapids-cmake_BINARY_DIR}")
  define_property(GLOBAL PROPERTY ${prefix}_populated)

  # construct a symlink from the source to the build dir
  # so we get the latest local changes without issue
  execute_process(
       COMMAND ${CMAKE_COMMAND} -E make_directory "${scratch_dir}")
  execute_process(
       COMMAND ${CMAKE_COMMAND} -E create_symlink "${local-rapids-cmake-root}" "${rapids-cmake_SOURCE_DIR}"
       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
       ECHO_OUTPUT_VARIABLE
       ECHO_ERROR_VARIABLE)
  unset(scratch_dir)

  message(STATUS "${local-rapids-cmake-root} -> ${rapids-cmake_SOURCE_DIR}")
  add_subdirectory(${rapids-cmake_SOURCE_DIR} ${rapids-cmake_BINARY_DIR})
endif()
