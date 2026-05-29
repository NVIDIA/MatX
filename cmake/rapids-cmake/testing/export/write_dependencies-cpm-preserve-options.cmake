# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)
include(${rapids-cmake-dir}/export/write_dependencies.cmake)

rapids_export_cpm(INSTALL RMM test_set CPM_ARGS NAME RMM VERSION 2.0 OPTIONS
                                                "FAKE_PACKAGE_ARGS FALSE"
                  GLOBAL_TARGETS RMM::RMM_POOL)

rapids_export_write_dependencies(install test_set "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake")

# Parse the `export_set.cmake` file for correct escaped args to `CPMFindPackage` calls
set(to_match_string [=["NAME;RMM;VERSION;2.0;OPTIONS;FAKE_PACKAGE_ARGS FALSE"]=])

file(READ "${CMAKE_CURRENT_BINARY_DIR}/export_set.cmake" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_export_write_dependencies(BUILD) failed to preserve quotes around CPM arguments"
  )
endif()
