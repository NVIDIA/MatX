# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)

project(fill_cache LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)

include(${rapids-cmake-dir}/cpm/cccl.cmake)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
include(${rapids-cmake-dir}/cpm/gbench.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)
include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
include(${rapids-cmake-dir}/cpm/nvtx3.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)

rapids_cpm_init(GENERATE_PINNED_VERSIONS)

# Download all source packages
set(CPM_DOWNLOAD_ALL "ON")
rapids_cpm_cccl(DOWNLOAD_ONLY ON)
rapids_cpm_cuco(DOWNLOAD_ONLY ON)
rapids_cpm_gbench(DOWNLOAD_ONLY ON)
rapids_cpm_gtest(DOWNLOAD_ONLY ON)
rapids_cpm_rapids_logger(DOWNLOAD_ONLY ON)
rapids_cpm_nvbench(DOWNLOAD_ONLY ON)
rapids_cpm_nvtx3(DOWNLOAD_ONLY ON)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

# Download all binary packages
set(CPM_DOWNLOAD_ALL "OFF")
rapids_cpm_nvcomp(USE_PROPRIETARY_BINARY ON DOWNLOAD_ONLY ON)

# Print out all the packages in the cache
get_cmake_property(cpm_packages CACHE_VARIABLES)
set(packages)
foreach(p ${cpm_packages})
  if("${p}" MATCHES "^CPM_PACKAGE_(.*)_SOURCE_DIR")
    list(APPEND packages ${CMAKE_MATCH_1})
  endif()
endforeach()
string(REPLACE ";" ", " packages "${packages}")
message(STATUS "CPM packages in cache: {${packages}}")
