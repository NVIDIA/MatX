# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cmake_parse_version
--------------------------

.. versionadded:: v21.06.00

Extract components of a `X.Y.Z` or `X.Y` version string in a consistent manner

  .. code-block:: cmake

    rapids_cmake_parse_version( [MAJOR|MINOR|PATCH|MAJOR_MINOR] version out_variable_name)

Offers the ability to extract components of any 2 or 3 component version string without
having to write complex regular expressions.

``MAJOR``
    Extract the first component (`X`) from `version` and place it in the variable
    named in `out_variable_name`

``MINOR``
    Extract the second component (`Y`) from `version` and place it in the variable
    named in `out_variable_name`

``PATCH``
    Extract the third component (`Z`) from `version` and place it in the variable
    named in `out_variable_name`. If no `Z` component exists for `version` nothing
    will happen.

``MAJOR_MINOR``
    Extract the first and second component (`X.Y`) from `version` and place it in the variable
    named in `out_variable_name` using the pattern `X.Y`.

Example on how to properly use :cmake:command:`rapids_cmake_parse_version`:

  .. code-block:: cmake

    project(Example VERSION 43.01.0)

    rapids_cmake_parse_version(MAJOR_MINOR ${PROJECT_VERSION} major_minor)
    message(STATUS "The major.minor version is: ${major_minor}")


Result Variables
^^^^^^^^^^^^^^^^
  The variable `out_variable_name` will be created/modified only when the version extraction
  is successful


#]=======================================================================]
function(rapids_cmake_parse_version mode version_value out_variable_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.parse_version")

  # target exists early terminate
  string(TOUPPER ${mode} mode)
  string(REPLACE "." ";" version_as_list "${version_value}")

  list(LENGTH version_as_list len)

  # Extract each component and make sure they aren't empty before setting. Enforces the rule that a
  # value/character must exist between each `.`
  if(mode STREQUAL "MAJOR" AND len GREATER_EQUAL 1)
    list(GET version_as_list 0 extracted_component)
    if(NOT extracted_component STREQUAL "")
      set(${out_variable_name} ${extracted_component} PARENT_SCOPE)
    endif()

  elseif(mode STREQUAL "MINOR" AND len GREATER_EQUAL 2)
    list(GET version_as_list 1 extracted_component)
    if(NOT extracted_component STREQUAL "")
      set(${out_variable_name} ${extracted_component} PARENT_SCOPE)
    endif()

  elseif(mode STREQUAL "PATCH" AND len GREATER_EQUAL 3)
    list(GET version_as_list 2 extracted_component)
    if(NOT extracted_component STREQUAL "")
      set(${out_variable_name} ${extracted_component} PARENT_SCOPE)
    endif()

  elseif(mode STREQUAL "MAJOR_MINOR" AND len GREATER_EQUAL 2)
    list(GET version_as_list 0 extracted_major)
    list(GET version_as_list 1 extracted_minor)
    if(NOT extracted_major STREQUAL "" AND NOT extracted_minor STREQUAL "")
      set(${out_variable_name} "${extracted_major}.${extracted_minor}" PARENT_SCOPE)
    endif()

  endif()
endfunction()
