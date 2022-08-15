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

add_library(real_target UNKNOWN IMPORTED)
set(targets fake_target1_ real_target fake_target2_ )
rapids_cmake_make_global(targets)

get_target_property(is_global real_target IMPORTED_GLOBAL)
if(NOT is_global)
  message(FATAL_ERROR "Expected is_global to be True [${is_global}]")
endif()
