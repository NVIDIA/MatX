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
cmake_minimum_required(VERSION 3.26.4)
project(rapids-cpm_find-patch-command-project LANGUAGES CXX)

include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(fmt version repository tag shallow exclude)

set(deps_dir "${CMAKE_CURRENT_BINARY_DIR}/_fmt_dep")
if(NOT EXISTS "${deps_dir}")
  file(MAKE_DIRECTORY "${deps_dir}")
  find_package(Git)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch "${tag}" "${repository}"
    WORKING_DIRECTORY "${deps_dir}")
endif()

set(fmt_dir "${deps_dir}/fmt")
list(APPEND CMAKE_PREFIX_PATH "${fmt_dir}")

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override(${CMAKE_CURRENT_SOURCE_DIR}/override.json)

include(${rapids-cmake-dir}/cpm/fmt.cmake)
rapids_cpm_fmt()

if(NOT "${fmt_ADDED}")
    message(FATAL_ERROR "The found repo was used rather than downloading and patching a new version")
endif()

# Verify that the two files that we inserted into the CCCL source tree exist
# Which proves the patches in the override are properly applied
if(NOT EXISTS "${fmt_SOURCE_DIR}/git_file_1.txt")
  message(FATAL_ERROR "failed to apply fmt first patch")
endif()

if(NOT EXISTS "${fmt_SOURCE_DIR}/git_file_2.txt")
  message(FATAL_ERROR "failed to apply fmt second patch")
endif()

execute_process(
    COMMAND ${GIT_EXECUTABLE} diff --quiet ${tag}
    RESULT_VARIABLE REPO_IS_PATCHED
    WORKING_DIRECTORY "${fmt_SOURCE_DIR}")

if(NOT ${REPO_IS_PATCHED})
    message(FATAL_ERROR "The repo was downloaded to ${fmt_SOURCE_DIR} but not patched.")
endif()
