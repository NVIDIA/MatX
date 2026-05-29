# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/nvtx3.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)

rapids_cpm_init(GENERATE_PINNED_VERSIONS)
rapids_cpm_gtest(DOWNLOAD_ONLY ON)
rapids_cpm_nvtx3(DOWNLOAD_ONLY ON)
rapids_cpm_rmm(DOWNLOAD_ONLY ON)

# only check projects that were downloaded by CPM (ignore those already in the build environment)
foreach(proj IN ITEMS GTest nvtx3 rmm)
  if(${proj}_SOURCE_DIR)
    list(APPEND projects-to-verify ${proj})
  endif()
endforeach()

include("${rapids-cmake-testing-dir}/cpm/verify_generated_pins.cmake")
verify_generated_pins(verify_pins PROJECTS ${projects-to-verify})
