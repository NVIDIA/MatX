#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/find/package.cmake)

set(CMAKE_PREFIX_PATH "${rapids-cmake-testing-dir}/find/find_package-components/")

rapids_find_package(FakeDependency 11 COMPONENTS AAAAA
                    BUILD_EXPORT_SET test_export_set
                    )

if(FakeDependency_FOUND)
  message(FATAL_ERROR "rapids_find_package recorded incorrect FOUND state for a failed find_package request")
endif()

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_FakeDependency.cmake")
if(EXISTS "${path}")
  message(FATAL_ERROR "rapids_find_package(BUILD) recorded a failed find_package request")
endif()
