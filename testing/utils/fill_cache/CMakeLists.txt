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
cmake_minimum_required(VERSION 3.20)

project(fill_cache LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)

include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
include(${rapids-cmake-dir}/cpm/spdlog.cmake)
include(${rapids-cmake-dir}/cpm/thrust.cmake)

rapids_cpm_init()

set(CPM_SOURCE_CACHE "${CMAKE_BINARY_DIR}")

# Download all source packages
set(CPM_DOWNLOAD_ALL "ON")
rapids_cpm_gtest(DOWNLOAD_ONLY ON)
rapids_cpm_libcudacxx(DOWNLOAD_ONLY ON)
rapids_cpm_nvbench(DOWNLOAD_ONLY ON)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)
rapids_cpm_spdlog(DOWNLOAD_ONLY ON)
rapids_cpm_thrust(temp DOWNLOAD_ONLY ON)
rapids_cpm_find(skbuild 0.14.1
                GIT_REPOSITORY https://github.com/scikit-build/scikit-build.git
                GIT_TAG 0.14.1
                )

# Download all binary packages
set(CPM_DOWNLOAD_ALL "OFF")
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON DOWNLOAD_ONLY ON)
