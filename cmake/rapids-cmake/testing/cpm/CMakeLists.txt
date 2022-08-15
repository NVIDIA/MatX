#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
add_cmake_config_test( rapids-cpm.cmake )

add_cmake_config_test( cpm_find-add-pkg-source )
add_cmake_config_test( cpm_find-and-find_package )
add_cmake_config_test( cpm_find-existing-build-dir )
add_cmake_config_test( cpm_find-existing-target )
add_cmake_config_test( cpm_find-existing-target-to-export-sets )
add_cmake_config_test( cpm_find-options-escaped )
add_cmake_config_test( cpm_find-restore-cpm-vars )
add_cmake_config_test( cpm_find-version-explicit-install.cmake )

add_cmake_config_test( cpm_init-bad-override-path.cmake SHOULD_FAIL "rapids_cpm_package_override can't load")
add_cmake_config_test( cpm_init-override-multiple.cmake )
add_cmake_config_test( cpm_init-override-simple.cmake )

add_cmake_config_test( cpm_package_override-bad-path.cmake SHOULD_FAIL "rapids_cpm_package_override can't load")
add_cmake_config_test( cpm_package_override-before-init.cmake )
add_cmake_config_test( cpm_package_override-empty.cmake )
add_cmake_config_test( cpm_package_override-multiple.cmake )
add_cmake_config_test( cpm_package_override-simple.cmake )

add_cmake_config_test( cpm_gtest-export.cmake )
add_cmake_config_test( cpm_gtest-simple.cmake )

add_cmake_config_test( cpm_libcudacxx-after_cpmfind.cmake )
add_cmake_config_test( cpm_libcudacxx-export.cmake )
add_cmake_config_test( cpm_libcudacxx-simple.cmake )

add_cmake_config_test( cpm_nvbench-export.cmake )
add_cmake_config_test( cpm_nvbench-simple.cmake )

add_cmake_config_test( cpm_nvcomp-export.cmake )
add_cmake_config_test( cpm_nvcomp-proprietary-off.cmake )
add_cmake_config_test( cpm_nvcomp-proprietary-on.cmake )
add_cmake_config_test( cpm_nvcomp-simple.cmake )
add_cmake_config_test( cpm_nvcomp-invalid-arch.cmake )
add_cmake_config_test( cpm_nvcomp-override-clears-proprietary_binary.cmake )

add_cmake_config_test( cpm_rmm-export.cmake )
add_cmake_config_test( cpm_rmm-simple.cmake )

add_cmake_config_test( cpm_spdlog-export.cmake )
add_cmake_config_test( cpm_spdlog-simple.cmake )

add_cmake_config_test( cpm_thrust-export.cmake )
add_cmake_config_test( cpm_thrust-simple.cmake )

add_cmake_config_test( cpm_cuco-simple.cmake )
add_cmake_config_test( cpm_cuco-export.cmake )
