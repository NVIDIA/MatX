# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/find.cmake)

cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

rapids_cpm_init()
rapids_cpm_find(RapidsTest 20.09.00
                GLOBAL_TARGETS RapidsTest::RapidsTest
                BUILD_EXPORT_SET example_export_set
                CPM_ARGS DOWNLOAD_COMMAND "ls" # Fake comment
                OPTIONS "BUILD_TESTS TRUE" "BUILD_EXAMPLES FALSE")
add_library(exampleLib INTERFACE)
install(TARGETS exampleLib EXPORT example_export_set)

include(${rapids-cmake-dir}/export/export.cmake)
rapids_export(BUILD rapids-test-project EXPORT_SET example_export_set)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-test-project-dependencies.cmake")
  message(FATAL_ERROR "rapids_cpm_find(BUILD) failed to generate a CPM configuration")
endif()

# We need to validate that we propagate NAME, VERSION, and CPM ARGS properly to the exported
# dependency file
set(to_match_string
    [=["NAME;RapidsTest;VERSION;20.09.00;DOWNLOAD_COMMAND;ls;OPTIONS;BUILD_TESTS TRUE;BUILD_EXAMPLES FALSE"]=]
)
file(READ "${CMAKE_BINARY_DIR}/rapids-test-project-dependencies.cmake" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_find(BUILD) failed to preserve quotes around CPM arguments")
endif()
