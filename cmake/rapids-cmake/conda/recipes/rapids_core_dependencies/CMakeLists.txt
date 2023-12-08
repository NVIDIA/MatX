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
cmake_minimum_required(VERSION 3.23.1)

project(rapids-dependencies LANGUAGES CXX)

set(rapids-cmake-dir "${PROJECT_SOURCE_DIR}/../../../rapids-cmake/")

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init()

# Download all source packages
include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
include(${rapids-cmake-dir}/cpm/thrust.cmake)
set(CPM_DOWNLOAD_ALL "ON")
rapids_cpm_libcudacxx(INSTALL_EXPORT_SET rapids-dependencies)
rapids_cpm_thrust(NAMESPACE temp INSTALL_EXPORT_SET rapids-dependencies)
