# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

# Fake out GIT_EXECUTABLE with a stub so that all git commands fail, acting like what happens when
# git doesn't exist
find_package(Git)
set(GIT_EXECUTABLE "/fake/path")
rapids_cmake_write_git_revision_file(git_generated_header
                                     "${CMAKE_CURRENT_BINARY_DIR}/demo/git_version.hpp")
