# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/build_type.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/download_with_retry.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/install_lib_dir.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/parse_version.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/support_conda_env.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/write_git_revision_file.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/write_version_file.cmake)
