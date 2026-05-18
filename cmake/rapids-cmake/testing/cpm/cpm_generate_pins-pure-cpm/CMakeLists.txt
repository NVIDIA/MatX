# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_init()
rapids_cpm_generate_pinned_versions(OUTPUT "${CMAKE_BINARY_DIR}/output_pinned_versions.json")

set(PKG_VERSION 0.6.2)
set(PKG_REPOSITORY https://github.com/nmslib/hnswlib.git)
rapids_cpm_find(hnswlib ${PKG_VERSION}
                GLOBAL_TARGETS hnswlib::hnswlib
                CPM_ARGS
                GIT_REPOSITORY ${PKG_REPOSITORY}
                GIT_TAG v${PKG_VERSION}
                GIT_SHALLOW TRUE
                DOWNLOAD_ONLY ON
                EXCLUDE_FROM_ALL ON)

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins PROJECTS hnswlib PIN_FILE
                      "${CMAKE_BINARY_DIR}/output_pinned_versions.json")
