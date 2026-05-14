# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

cmake_path(GET rapids-cmake-dir PARENT_PATH rapids-git-dir)
cmake_path(APPEND rapids-git-dir ".git")

# Remove this cache variable so we can properly test fetching rapids-cmake work as intended
unset(rapids-cmake-dir CACHE)

set(CMAKE_MODULE_PATH "/not/a/directory")
add_subdirectory(A)
add_subdirectory(B)
