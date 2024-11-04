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
include(${rapids-cmake-dir}/cpm/rmm.cmake)

set(RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE "${CMAKE_CURRENT_BINARY_DIR}/custom_path/pinned_versions.json")
rapids_cpm_init(GENERATE_PINNED_VERSIONS)
rapids_cpm_rmm()

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins_A
    PROJECTS rmm fmt spdlog
    PIN_FILE "${RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE}")

verify_generated_pins(verify_pins_B
    PROJECTS rmm fmt spdlog
    PIN_FILE "${CMAKE_BINARY_DIR}/rapids-cmake/pinned_versions.json")
