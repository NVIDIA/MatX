# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)

rapids_export_cpm(build FAKE_CPM_PACKAGE test_export_set
                  CPM_ARGS NAME FAKE_CPM_PACKAGE VERSION 1.0 OPTIONS "FAKE_PACKAGE_OPTION_A TRUE"
                           "FAKE_PACKAGE_OPTION_B FALSE")

rapids_export_cpm(install FAKE_CPM_PACKAGE test_export_set
                  CPM_ARGS NAME FAKE_CPM_PACKAGE VERSION 1.0 OPTIONS "FAKE_PACKAGE_OPTION_A TRUE"
                           "FAKE_PACKAGE_OPTION_B FALSE")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake")
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to generate a CPM configuration")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/cpm_FAKE_CPM_PACKAGE.cmake")
  message(FATAL_ERROR "rapids_export_cpm(INSTALL) failed to generate a CPM configuration")
endif()

# We need to validate both of the files all CPM args in quotes
#
set(to_match_string
    [=["NAME;FAKE_CPM_PACKAGE;VERSION;1.0;OPTIONS;FAKE_PACKAGE_OPTION_A TRUE;FAKE_PACKAGE_OPTION_B FALSE"]=]
)

file(READ "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/cpm_FAKE_CPM_PACKAGE.cmake"
     contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(BUILD) failed to preserve quotes around CPM arguments")
endif()

file(READ "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/cpm_FAKE_CPM_PACKAGE.cmake"
     contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_cpm(INSTALL) failed to preserve quotes around CPM arguments")
endif()
