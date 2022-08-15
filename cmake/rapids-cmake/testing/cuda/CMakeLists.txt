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
add_cmake_config_test( rapids-cuda.cmake )

add_cmake_config_test( init_runtime-existing.cmake )
add_cmake_config_test( init_runtime-multiple.cmake )
add_cmake_config_test( init_runtime-shared.cmake )
add_cmake_config_test( init_runtime-static.cmake )

add_cmake_config_test( init_arch-all-via-undef )
add_cmake_config_test( init_arch-all.cmake )
add_cmake_config_test( init_arch-existing-project-flags.cmake )
add_cmake_config_test( init_arch-native.cmake )
add_cmake_config_test( init_arch-native-via-empty-str )
add_cmake_config_test( init_arch-user.cmake )

add_cmake_config_test( set_arch-all.cmake )
add_cmake_config_test( set_arch-existing.cmake )
add_cmake_config_test( set_arch-invalid-mode.cmake )
add_cmake_config_test( set_arch-native.cmake )

