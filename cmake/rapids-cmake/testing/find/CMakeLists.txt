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
add_cmake_config_test( rapids-find.cmake )

add_cmake_config_test( find_package-no-variable-leak.cmake )

add_cmake_config_test( find_package-build.cmake )
add_cmake_config_test( find_package-install.cmake )

# The inverse of `find_package-optional-failed` is the
# above two tests.
# The inverse of `find_package-required-found` is always
# a hard error and therefore has no test
add_cmake_config_test( find_package-optional-failed.cmake )
add_cmake_config_test( find_package-required-found.cmake )
add_cmake_config_test( find_package-version-explicit-failed.cmake )

add_cmake_config_test( find_package-version-explicit.cmake )
add_cmake_config_test( find_package-version-none.cmake )
add_cmake_config_test( find_package-version-required-explicit-failed.cmake SHOULD_FAIL "but required is exact version \"99999999999\"")

add_cmake_config_test( find_generate_module-build )
add_cmake_config_test( find_generate_module-install )
add_cmake_config_test( find_generate_module-header-only )
