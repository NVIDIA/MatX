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
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/export.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

cmake_minimum_required(VERSION 3.23.1)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

rapids_export_cpm(BUILD RaFT FakEProJecT
                  CPM_ARGS
                    FAKE_PACKAGE_ARGS TRUE
                  )

rapids_export_write_dependencies(BUILD FakEProJecT dependencies.cmake)

rapids_export(BUILD FakEProJecT
  EXPORT_SET fake_set
  NAMESPACE test::
  )
rapids_export(INSTALL FakEProJecT
  EXPORT_SET fake_set
  NAMESPACE test::
  )

check_copyright_header("${CMAKE_BINARY_DIR}/fakeproject-config.cmake")
check_copyright_header("${CMAKE_BINARY_DIR}/rapids-cmake/fakeproject/export/fakeproject/fakeproject-config.cmake")
check_copyright_header("${CMAKE_BINARY_DIR}/rapids-cmake/FakEProJecT/build/cpm_RaFT.cmake")
check_copyright_header("${CMAKE_BINARY_DIR}/dependencies.cmake")
