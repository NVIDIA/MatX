# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
#
# This is the legacy entry point for projects using rapids-cmake.
#
# cmake-format: off
# This will setup the following variables in the parent directory
# - CMAKE_MODULE_PATH
# - rapids-cmake-dir
# cmake-format: on
#
# This is considered legacy as it has issues when multiple projects use rapids-cmake via CPM inside
# the same global project. In those cases it can fail due to CMAKE_MODULE_PREFIX not being exported
# properly

# Enforce the minimum required CMake version for all users
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

set(rapids-cmake-dir "${CMAKE_CURRENT_LIST_DIR}/rapids-cmake")
if(NOT DEFINED CACHE{rapids-cmake-dir})
  set(rapids-cmake-dir "${rapids-cmake-dir}" CACHE INTERNAL "" FORCE)
endif()

if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
endif()

# Propagate up the rapids-cmake version/tag info
include("${rapids-cmake-dir}/rapids-version.cmake")
set(rapids-cmake-version ${rapids-cmake-version} PARENT_SCOPE)
set(rapids-cmake-checkout-tag ${rapids-cmake-checkout-tag} PARENT_SCOPE)

# install a hook that sets up `rapids-cmake-dir` and `CMAKE_MODULE_PATH` all the way up the
# call-stack
cmake_language(DEFER DIRECTORY ${CMAKE_CURRENT_LIST_DIR} CALL include
               "${rapids-cmake-dir}/../init.cmake")
