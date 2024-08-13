#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

add_cmake_config_test(rapids-cython-core.cmake)

add_cmake_config_test(cython-core_init.cmake)
add_cmake_config_test(cython-core_create_modules_errors.cmake SHOULD_FAIL "You must call rapids_cython_init before calling this function")

add_cmake_build_test(cython-core_create_modules)
add_cmake_config_test(cython-core_create_modules_with_library)
add_cmake_config_test(cython-core_create_modules_with_prefix)

add_cmake_config_test(cython-core_add_rpath_entries)
