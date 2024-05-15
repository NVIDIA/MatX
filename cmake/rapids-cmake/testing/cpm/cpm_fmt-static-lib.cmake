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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/fmt.cmake)

enable_language(CXX)

rapids_cpm_init()

set(CMAKE_BUILD_SHARED_LIBS OFF)
rapids_cpm_fmt()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/use_fmt.cpp" [=[
#include <fmt/core.h>

std::string make_error_string() {
  std::string expect = fmt::format("The answer is {:d}", "forty-two");
  return expect;
}
]=])

add_library(uses_fmt SHARED "${CMAKE_CURRENT_BINARY_DIR}/use_fmt.cpp")
target_link_libraries(uses_fmt PRIVATE fmt::fmt)
