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
include(${rapids-cmake-dir}/cython/add_rpath_entries.cmake)

project(rapids_cython-set_lib_dirs LANGUAGES C CXX)

# Silence warning about running without scikit-build.
set(SKBUILD ON)

# Ensure that scikit-build's CMake files are discoverable. The glob is to
# capture the current git commit hash.
file(GLOB skbuild_resource_dir LIST_DIRECTORIES ON "${CPM_SOURCE_CACHE}/skbuild/*/skbuild/resources/cmake")
LIST(APPEND CMAKE_MODULE_PATH "${skbuild_resource_dir}")

rapids_cython_init()

rapids_cython_create_modules(
    SOURCE_FILES test.pyx
    ASSOCIATED_TARGETS associated_target
    )

# Test a relative path.
rapids_cython_add_rpath_entries(TARGET associated_target PATHS ../libraries)
get_target_property(rpath test INSTALL_RPATH)
if (NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()

# Test a relative path with a root directory.
rapids_cython_add_rpath_entries(TARGET associated_target PATHS ../libraries ROOT_DIRECTORY ../)
get_target_property(rpath test INSTALL_RPATH)
if (NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries;$ORIGIN/../../libraries")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()

# Test an absolute path
rapids_cython_add_rpath_entries(TARGET associated_target PATHS /path/to/libraries)
get_target_property(rpath test INSTALL_RPATH)
set(the_path /path/to/libraries)
cmake_path(RELATIVE_PATH the_path)
if (NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries;$ORIGIN/../../libraries;$ORIGIN/${the_path}")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()
