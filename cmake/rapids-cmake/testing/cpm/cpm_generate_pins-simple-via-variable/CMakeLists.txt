# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

set(RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/custom_path/pinned_versions.json")

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gbench.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

rapids_cpm_init()
rapids_cpm_gbench()
rapids_cpm_gtest()

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins PROJECTS benchmark GTest PIN_FILE
                      "${RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE}")
