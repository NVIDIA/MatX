# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)

rapids_cpm_init()
rapids_cpm_package_override("${CMAKE_CURRENT_BINARY_DIR}/../rapids-cmake/pinned_versions.json")

# Verify that each git_tag is now different.
rapids_cpm_package_details_internal(rmm rmm_version rmm_repository pin_rmm_tag rmm_src_dir
                                    rmm_shallow rmm_exclude)

if(NOT pin_rmm_tag STREQUAL "e3e32150d1bef13480ede4b2df1f0b38e43108a4")
  message(FATAL_ERROR "pinned rmm tag (${pin_rmm_tag}) doesn't match pre-computed SHA1 for the 22.02 tag"
  )
endif()
