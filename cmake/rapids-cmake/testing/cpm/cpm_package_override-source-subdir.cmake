# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

# Need to write out a default file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/default.json
     [=[
{
  "packages": {
    "zstd": {
      "version": "1.5.7",
      "git_tag": "v${version}",
      "source_subdir" : "build/cmake"
    }
  }
}
  ]=])

rapids_cpm_init(CUSTOM_DEFAULT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/default.json")

set(CPM_DOWNLOAD_ALL ON) # required so we don't find a local installed version of zstd
rapids_cpm_find(zstd 1.5.7
                GLOBAL_TARGETS zstd
                CPM_ARGS
                GIT_REPOSITORY https://github.com/facebook/zstd.git
                GIT_TAG v1.5.7
                GIT_SHALLOW FALSE SOURCE_SUBDIR build/cmake
                OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
                        "ZSTD_BUILD_PROGRAMS OFF" "BUILD_SHARED_LIBS OFF")

if(NOT TARGET libzstd_static)
  message(FATAL_ERROR "source_subdir in override was ignored for `rapids_cpm_find`")
endif()
