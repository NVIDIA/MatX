# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

add_subdirectory(a)
add_subdirectory(b)

# Turn on pin generation after cpm dependencies have been added to make sure it is order invariant
include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init(GENERATE_PINNED_VERSIONS)

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins PROJECTS benchmark GTest)
