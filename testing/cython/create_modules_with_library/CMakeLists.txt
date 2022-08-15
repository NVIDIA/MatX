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

cmake_minimum_required(VERSION 3.20)

include(${rapids-cmake-dir}/cython/create_modules.cmake)
include(${rapids-cmake-dir}/cython/init.cmake)

project(rapids_cython-create_modules LANGUAGES C CXX)

# Silence warning about running without scikit-build.
set(SKBUILD ON)

# Ensure that scikit-build's CMake files are discoverable. The glob is to
# capture the current git commit hash.
file(GLOB skbuild_resource_dir LIST_DIRECTORIES ON "${CPM_SOURCE_CACHE}/skbuild/*/skbuild/resources/cmake")
LIST(APPEND CMAKE_MODULE_PATH "${skbuild_resource_dir}")

rapids_cython_init()

# Test that we can specify a (fake) library.
add_library(rapids_cython_test_library test)
rapids_cython_create_modules(
    SOURCE_FILES test.pyx
    LINKED_LIBRARIES rapids_cython_test_library
    )
