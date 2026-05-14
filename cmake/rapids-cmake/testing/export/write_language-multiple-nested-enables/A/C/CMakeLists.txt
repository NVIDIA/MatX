# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include(${rapids-cmake-dir}/export/write_language.cmake)

function(enable_nested_language file_path)
  # replicates CPM including the file
  include("${file_path}")
endfunction()

set(path "${CMAKE_CURRENT_BINARY_DIR}/enable_cuda_language.cmake")
rapids_export_write_language(BUILD CXX "${path}")
enable_nested_language("${path}")

add_library(C STATIC static.cpp)
