# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/install_lib_dir.cmake)

unset(ENV{CONDA_BUILD})
unset(ENV{CONDA_PREFIX})

include(GNUInstallDirs)
set(old_CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})

rapids_cmake_install_lib_dir(lib_dir)
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
  message(FATAL_ERROR "rapids_cmake_install_lib_dir shouldn't have caused the CMAKE_INSTALL_LIBDIR variable to not exist"
  )
endif()

if(NOT lib_dir STREQUAL CMAKE_INSTALL_LIBDIR)
  message(FATAL_ERROR "rapids_cmake_install_lib_dir computed '${lib_dir}', but we expected '${CMAKE_INSTALL_LIBDIR}' as it should match GNUInstallDirs"
  )
endif()

if(NOT lib_dir STREQUAL old_CMAKE_INSTALL_LIBDIR)
  message(FATAL_ERROR "rapids_cmake_install_lib_dir computed '${lib_dir}', but we expected '${CMAKE_INSTALL_LIBDIR}' as it should match GNUInstallDirs"
  )
endif()

# unset CMAKE_INSTALL_LIBDIR so it doesn't leak into our CMakeCache.txt and cause subsequent re-runs
# of the test to fail
unset(CMAKE_INSTALL_LIBDIR)
unset(CMAKE_INSTALL_LIBDIR CACHE)
