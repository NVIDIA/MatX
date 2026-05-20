# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cmake_build_type
-----------------------

.. versionadded:: v21.06.00

Establish the :cmake:variable:`CMAKE_BUILD_TYPE <cmake:variable:CMAKE_BUILD_TYPE>` default value.

  .. code-block:: cmake

    rapids_cmake_build_type(default_type)

If the generator is `Ninja` or `Makefile` the :cmake:variable:`CMAKE_BUILD_TYPE <cmake:variable:CMAKE_BUILD_TYPE>`
variable will be established if not explicitly set by the user either by
the env variable `CMAKE_BUILD_TYPE` or by passing `-DCMAKE_BUILD_TYPE=`. This removes
situations where the `No-Config` / `Empty` build type is used.

``default_type``
  The default build type to use if one doesn't already exist

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`CMAKE_BUILD_TYPE <cmake:variable:CMAKE_BUILD_TYPE>` will be set to ``default_type`` if not already set

#]=======================================================================]
function(rapids_cmake_build_type default_type)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.build_type")

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(VERBOSE "Setting build type to '${default_type}' since none specified.")
    set(CMAKE_BUILD_TYPE "${default_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                                 "RelWithDebInfo")
  endif()
endfunction()
