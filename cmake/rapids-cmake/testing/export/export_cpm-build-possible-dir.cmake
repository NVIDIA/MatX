# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)

# Verify valid dir is picked up
set(FAKE_CPM_PACKAGE_DIR "/valid/looking/path")
rapids_export_cpm(build FAKE_CPM_PACKAGE test_export_set CPM_ARGS FAKE_PACKAGE_ARGS TRUE)

# Verify that cpm configuration files exist
set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake")
if(NOT EXISTS "${path}")
  message(FATAL_ERROR "rapids_export_cpm failed to generate a CPM configuration")
endif()

# verify that the expected path exists in FAKE_CPM_PACKAGE.cmake
set(to_match_string [=[set(possible_package_dir "/valid/looking/path")]=])
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to write out the possible_package_dir")
endif()

# Verify in-valid dir is ignored
set(also_fake_cpm_package_DIR OFF)
set(also_fake_cpm_package_BINARY_DIR /binary/dir/path/)
rapids_export_cpm(BUILD also_fake_cpm_package test_export_set CPM_ARGS VERSION 2.0
                  GLOBAL_TARGETS ABC::ABC ABC::CBA)
set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_also_fake_cpm_package.cmake")
if(NOT EXISTS "${path}")
  message(FATAL_ERROR "rapids_export_cpm failed to generate a CPM configuration")
endif()

# verify that the expected path exists in also_fake_cpm_package.cmake
set(to_match_string [=[set(possible_package_dir "/binary/dir/path/")]=])
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to write out the possible_package_dir")
endif()
