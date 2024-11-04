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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

rapids_cpm_init()
rapids_cpm_bs_thread_pool(BUILD_EXPORT_SET test)

# Add a custom command that verifies that the expect files have
# been installed for each component
file(WRITE "${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.26.4)
project(verify_bs_thread_pool LANGUAGES CXX)

set(CMAKE_PREFIX_PATH \"${CMAKE_BINARY_DIR}/\")
find_package(bs_thread_pool REQUIRED)

file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/stub.cpp\" \"#include <BS_thread_pool.hpp>\")
add_library(uses_bs_thread_pool SHARED stub.cpp)
target_link_libraries(uses_bs_thread_pool PRIVATE BS::thread_pool)
")

add_custom_target(verify_build_config ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir/build"
  COMMAND ${CMAKE_COMMAND} -S="${CMAKE_BINARY_DIR}/check_bs_thread_pool_dir" -B="${CMAKE_BINARY_DIR}/build"
  COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/build"
)
