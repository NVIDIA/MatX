# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

rapids_cpm_init()
rapids_cpm_generate_pinned_versions(OUTPUT "${CMAKE_BINARY_DIR}/output_pinned_versions.json")

rapids_cpm_nvtx3(INSTALL_EXPORT_SET test)
set(CPM_DOWNLOAD_ALL ON) # required so we don't find a local installed version of zstd
rapids_cpm_find(zstd 1.5.7
                GLOBAL_TARGETS zstd
                CPM_ARGS
                GIT_REPOSITORY https://github.com/facebook/zstd.git
                GIT_TAG v1.5.7
                GIT_SHALLOW FALSE SOURCE_SUBDIR build/cmake
                OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
                        "ZSTD_BUILD_PROGRAMS OFF" "BUILD_SHARED_LIBS OFF"
                INSTALL_EXPORT_SET test)

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins PROJECTS nvtx3 zstd SOURCE_SUBDIR_PROJECTS nvtx3 zstd PIN_FILE
                      "${CMAKE_BINARY_DIR}/output_pinned_versions.json")
