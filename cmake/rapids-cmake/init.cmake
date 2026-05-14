# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
#
# This is NOT an entry point for other projects using rapids-cmake
#
# Nothing but rapids-cmake/CMakeLists.txt should include this file
#
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)

  # Be defensive of other projects over-writing CMAKE_MODULE_PATH on us!
  set(rapids-cmake-dir "${rapids-cmake-dir}" PARENT_SCOPE)
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

  # Don't install this hook if another rapids project has already done so
  get_directory_property(parent_dir PARENT_DIRECTORY)
  cmake_language(DEFER DIRECTORY "${parent_dir}" GET_CALL_IDS rapids_existing_calls)
  if(NOT rapids_init_hook IN_LIST rapids_existing_calls)
    cmake_language(DEFER DIRECTORY "${parent_dir}" ID rapids_init_hook CALL include
                   "${rapids-cmake-dir}/../init.cmake")
  endif()
endif()
