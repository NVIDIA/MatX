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
include(${rapids-cmake-dir}/cmake/make_global.cmake)

add_library(test_lib UNKNOWN IMPORTED)
add_library(test::lib ALIAS test_lib)

set(targets test::lib)
rapids_cmake_make_global(targets)

# verify that test::lib AND test_lib are now global
get_target_property(alias_is_global test::lib ALIAS_GLOBAL)
if(alias_is_global)
  message(FATAL_ERROR "Expected alias_is_global to be False [${alias_is_global}]")
endif()
