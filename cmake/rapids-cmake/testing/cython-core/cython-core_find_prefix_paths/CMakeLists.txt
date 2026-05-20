# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.31)

include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)

project(rapids_cython-find_prefix_path_test LANGUAGES C)

find_package(Python REQUIRED COMPONENTS Interpreter)

set(env_dir "${CMAKE_CURRENT_BINARY_DIR}/env")
file(REMOVE_RECURSE "${env_dir}")
execute_process(COMMAND ${Python_EXECUTABLE} -m "venv" "${env_dir}")

set(Python_EXECUTABLE "${env_dir}/bin/python")

execute_process(COMMAND "${Python_EXECUTABLE}" "-m" "pip" "install" "libucx-cu13")
rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" prefix_paths)

set(_get_platlib
    [=[
import sysconfig
print(sysconfig.get_path("platlib"))
]=])

execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "${_get_platlib}"
                OUTPUT_VARIABLE site_packages_path)
string(STRIP "${site_packages_path}" site_packages_path)
set(libucx_path "${site_packages_path}/libucx")

if(NOT libucx_path IN_LIST prefix_paths)
  message(FATAL_ERROR "Expected to find libucx path '${libucx_path}' in prefix paths but did not.")
endif()
