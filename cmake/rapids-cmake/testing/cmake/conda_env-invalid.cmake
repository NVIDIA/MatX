# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/support_conda_env.cmake)

unset(ENV{CONDA_BUILD})
unset(ENV{CONDA_PREFIX})
set(ENV{BUILD_PREFIX} "/usr/local/build_prefix")
set(ENV{PREFIX} "/opt/local/prefix")

rapids_cmake_support_conda_env(conda_env)

if(TARGET conda_env)
  message(FATAL_ERROR "Not expected for `conda_env` target to exist")
endif()

set(before_call_value "${CMAKE_PREFIX_PATH}")
rapids_cmake_support_conda_env(conda_env2 MODIFY_PREFIX_PATH)
if(TARGET conda_env2)
  message(FATAL_ERROR "Not expected for `conda_env2` target to exist")
endif()

if(NOT "${before_call_value}" STREQUAL "${CMAKE_PREFIX_PATH}")
  message(FATAL_ERROR "Not expected for `rapids_cmake_support_conda_env` to modify CMAKE_PREFIX_PATH"
  )
endif()
