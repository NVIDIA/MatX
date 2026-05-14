# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.20)

include(${rapids-cmake-dir}/cython-core/create_modules.cmake)
include(${rapids-cmake-dir}/cython-core/init.cmake)
include(${rapids-cmake-dir}/cython-core/add_rpath_entries.cmake)

project(rapids_cython-set_lib_dirs LANGUAGES C CXX)

rapids_cython_init()

rapids_cython_create_modules(SOURCE_FILES test.pyx ASSOCIATED_TARGETS associated_target)

# Test a relative path.
rapids_cython_add_rpath_entries(TARGET associated_target PATHS ../libraries)
get_target_property(rpath test INSTALL_RPATH)
if(NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()

# Test a relative path with a root directory.
rapids_cython_add_rpath_entries(TARGET associated_target PATHS ../libraries ROOT_DIRECTORY ../)
get_target_property(rpath test INSTALL_RPATH)
if(NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries;$ORIGIN/../../libraries")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()

# Test an absolute path
rapids_cython_add_rpath_entries(TARGET associated_target PATHS /path/to/libraries)
get_target_property(rpath test INSTALL_RPATH)
set(the_path /path/to/libraries)
cmake_path(RELATIVE_PATH the_path)
if(NOT rpath STREQUAL "$ORIGIN;$ORIGIN/../libraries;$ORIGIN/../../libraries;$ORIGIN/${the_path}")
  message(FATAL_ERROR "rapids_cython_set_lib_dirs failed to set the RPATH correctly.")
endif()
