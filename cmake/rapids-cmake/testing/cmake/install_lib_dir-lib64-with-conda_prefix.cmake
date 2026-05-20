# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/install_lib_dir.cmake)

unset(ENV{CONDA_BUILD})
unset(ENV{CONDA_PREFIX})

set(ENV{CONDA_PREFIX} "/opt/conda/prefix")
set(CMAKE_INSTALL_PREFIX "/opt/not-conda/prefix")
set(CMAKE_INSTALL_LIBDIR "lib64")

rapids_cmake_install_lib_dir(lib_dir)

if(NOT lib_dir STREQUAL "lib64")
  message(FATAL_ERROR "rapids_cmake_install_lib_dir computed '${lib_dir}', but we expected 'lib64'")
endif()

# verify CMAKE_INSTALL_LIBDIR doesn't exist
if(NOT CMAKE_INSTALL_LIBDIR STREQUAL "lib64")
  message(FATAL_ERROR "CMAKE_INSTALL_LIBDIR now set to '${CMAKE_INSTALL_LIBDIR}', but we expected 'lib64'"
  )
endif()

set(CMAKE_INSTALL_PREFIX "/opt/conda/prefix")
unset(CMAKE_INSTALL_LIBDIR)
unset(CMAKE_INSTALL_LIBDIR CACHE)

rapids_cmake_install_lib_dir(lib_dir MODIFY_INSTALL_LIBDIR)

if(NOT lib_dir STREQUAL "lib")
  message(FATAL_ERROR "rapids_cmake_install_lib_dir computed '${lib_dir}', but we expected 'lib'")
endif()

# verify CMAKE_INSTALL_LIBDIR doesn't exist
if(NOT CMAKE_INSTALL_LIBDIR STREQUAL "lib")
  message(FATAL_ERROR "CMAKE_INSTALL_LIBDIR now set to '${CMAKE_INSTALL_LIBDIR}', but we expected 'lib'"
  )
endif()

# unset CMAKE_INSTALL_LIBDIR so it doesn't leak into our CMakeCache.txt and cause subsequent re-runs
# of the test to fail
unset(CMAKE_INSTALL_LIBDIR)
unset(CMAKE_INSTALL_LIBDIR CACHE)
