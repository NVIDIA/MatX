# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Emulate the variables and properties that FetchContent would set so tests that themselves download
# rapids-cmake will use the version we have symlinked.
include(FetchContent)

set(prefix "_FetchContent_rapids-cmake")
get_property(_rapids_already_hooked GLOBAL PROPERTY ${prefix}_populated DEFINED)
if(NOT _rapids_already_hooked)
  set(local-rapids-cmake-root "${rapids-cmake-dir}/..")
  cmake_path(NORMAL_PATH local-rapids-cmake-root)

  set(scratch_dir "${CMAKE_CURRENT_BINARY_DIR}/_deps")

  set(rapids-cmake_SOURCE_DIR "${scratch_dir}/rapids-cmake-src")
  set(rapids-cmake_BINARY_DIR "${scratch_dir}/rapids-cmake-build")
  set(rapids-cmake_POPULATED TRUE)
  set_property(GLOBAL PROPERTY ${prefix}_sourceDir "${rapids-cmake_SOURCE_DIR}")
  set_property(GLOBAL PROPERTY ${prefix}_binaryDir "${rapids-cmake_BINARY_DIR}")
  define_property(GLOBAL PROPERTY ${prefix}_populated)

  # construct a symlink from the source to the build dir so we get the latest local changes without
  # issue
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${scratch_dir}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${local-rapids-cmake-root}"
                          "${rapids-cmake_SOURCE_DIR}"
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" ECHO_OUTPUT_VARIABLE
                                    ECHO_ERROR_VARIABLE)
  unset(scratch_dir)

  message(STATUS "${local-rapids-cmake-root} -> ${rapids-cmake_SOURCE_DIR}")
  add_subdirectory(${rapids-cmake_SOURCE_DIR} ${rapids-cmake_BINARY_DIR})
endif()
