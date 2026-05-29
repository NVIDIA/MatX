# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include("${rapids-cmake-dir}/rapids-version.cmake")
cmake_path(GET rapids-cmake-dir PARENT_PATH rapids.cmake-location)
cmake_path(APPEND rapids.cmake-location "RAPIDS.cmake")

# Remove this cache variable so we can properly test fetching rapids-cmake work as intended
unset(rapids-cmake-dir CACHE)

add_subdirectory(A)
add_subdirectory(B)
