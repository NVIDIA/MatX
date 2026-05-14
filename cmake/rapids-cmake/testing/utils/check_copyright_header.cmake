# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

function(check_copyright_header file)
  cmake_path(GET file EXTENSION LAST_ONLY file_ext)
  string(TIMESTAMP current_year "%Y" UTC)
  if(file_ext STREQUAL ".txt" OR file_ext STREQUAL ".cmake")
    string(CONFIGURE [=[#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) @current_year@, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================
]=]
                     expected_header
           @ONLY)
  else()
    string(CONFIGURE [=[/*
 * SPDX-FileCopyrightText: Copyright (c) @current_year@, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
]=]
                     expected_header
           @ONLY)
  endif()
  string(LENGTH "${expected_header}" expected_header_length)

  file(READ "${file}" actual_header LIMIT "${expected_header_length}")
  if(NOT actual_header STREQUAL expected_header)
    message(FATAL_ERROR "File ${file} did not have expected copyright header")
  endif()
endfunction()
