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
cmake_minimum_required(VERSION 3.26.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/find/package.cmake)
rapids_cpm_init()
rapids_cpm_generate_pinned_versions(OUTPUT "${CMAKE_BINARY_DIR}/output_pinned_versions.json")

set(PKG_VERSION  0.6.2)
set(PKG_REPOSITORY https://github.com/nmslib/hnswlib.git)
rapids_cpm_find(
    hnswlib ${PKG_VERSION}
    GLOBAL_TARGETS hnswlib::hnswlib
    CPM_ARGS
    GIT_REPOSITORY ${PKG_REPOSITORY}
    GIT_TAG v${PKG_VERSION}
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY ON
    EXCLUDE_FROM_ALL ON
  )

# rapids_find_package should never be in CPM packages, but lets double check
rapids_find_package(Threads REQUIRED)

# rapids_cpm_find for found system installed packages will be in CPM package
# So we need to verify that we don't generate a pinned versions entry
# for this package
rapids_cpm_find(ZLIB 1.0 REQUIRED)


include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins
    PROJECTS hnswlib
    PROJECTS_NOT_EXIST ZLIB Threads
    PIN_FILE "${CMAKE_BINARY_DIR}/output_pinned_versions.json")
