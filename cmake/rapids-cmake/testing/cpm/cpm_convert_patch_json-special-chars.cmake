# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/detail/convert_patch_json.cmake)

set(bug
    [=[#include "file.h"
int function(not_parsed[
N], properly ) {
}]=])
set(file_path "${CMAKE_BINARY_DIR}/bug.txt")
file(WRITE ${file_path} "${bug}")

set(expected_output
    [==[[
"#include \"file.h\"",
"int function(not_parsed[",
"N], properly ) {",
"}"
]]==])

rapids_cpm_convert_patch_json(FROM_FILE_TO_JSON json FILE_VAR file_path)
string(JSON json_content GET "${json}" "content")

string(JSON content_length LENGTH "${json_content}")
math(EXPR content_length "${content_length} - 1")
foreach(index RANGE ${content_length})
  string(JSON computed_line GET "${json_content}" ${index})
  string(JSON expected_line GET "${expected_output}" ${index})
  if(NOT (computed_line STREQUAL expected_line))
    message(FATAL_ERROR "exp: `${expected_line}`\ngot: `${computed_line}`")
  endif()
endforeach()
