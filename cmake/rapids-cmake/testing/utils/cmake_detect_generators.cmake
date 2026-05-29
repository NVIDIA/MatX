# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
cmake_detect_generators
-----------------------
.. versionadded:: v21.06.00

Reports back what generators from the set of below are usable on the current machine
  - `Ninja`
  - `Ninja Multi-Config`
  - `Unix Makefiles`

#]=======================================================================]
function(cmake_detect_generators generator_name_var fancy_name_var)

  # See CMakeUnixFindMake and CMakeNinjaFindMake
  find_program(RAPIDS_TESTING_MAKE_GEN NAMES gmake make smake NAMES_PER_DIR)
  find_program(RAPIDS_TESTING_NINJA_GEN NAMES ninja-build ninja samu NAMES_PER_DIR)

  if(RAPIDS_TESTING_MAKE_GEN)
    list(APPEND supported_gens "Unix Makefiles")
    list(APPEND nice_names "makefile")
  endif()

  if(RAPIDS_TESTING_NINJA_GEN)
    list(APPEND supported_gens "Ninja")
    list(APPEND nice_names "ninja")

    list(APPEND supported_gens "Ninja Multi-Config")
    list(APPEND nice_names "ninja_multi-config")
  endif()
  set(${generator_name_var} "${supported_gens}" PARENT_SCOPE)
  set(${fancy_name_var} "${nice_names}" PARENT_SCOPE)
endfunction()
