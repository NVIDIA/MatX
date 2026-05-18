# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_export_post_find_package_code
------------------------------------

.. versionadded:: v23.12.00

Record for <PackageName> a set of CMake instructions to be executed after the package
has been found successfully.

.. code-block:: cmake

  rapids_export_post_find_package_code((BUILD|INSTALL)
                                       <PackageName>
                                       <code>
                                       EXPORT_SET [ExportSetName]
                                       [CONDITION <variableName>]
                                       )

When using complicated find modules like `Thrust` you might need to run some code after
execution. Multiple calls to :cmake:command:`rapids_export_post_find_package_code` will append the
instructions to execute in call order.

.. note:
  The code will only be run if the package was found

``BUILD``
  Record code to be executed immediately after `PackageName` has been found
  for our our build directory export set.

``INSTALL``
  Record code to be executed immediately after `PackageName` has been found
  for our our install directory export set.

``EXPORT_SET``
  List the export set name that this code should be attached too. If
  no name is given the associated call will be ignored.

``CONDITION``
  A boolean variable name, that when evaluates to undefined or a false value
  will cause the associated call to be ignored.

#]=======================================================================]
function(rapids_export_post_find_package_code type name code)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.post_find_package_code")

  set(options "")
  set(one_value EXPORT_SET CONDITION)
  set(multi_value "")
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
  # Early terminate conditions
  if(NOT _RAPIDS_EXPORT_SET OR NOT ${_RAPIDS_CONDITION})
    return()
  endif()

  string(TOLOWER ${type} type)
  set(export_set ${_RAPIDS_EXPORT_SET})

  if(NOT TARGET rapids_export_${type}_${export_set})
    add_library(rapids_export_${type}_${export_set} INTERFACE)
  endif()

  # if the code coming in is a list of string we will have `;`, so transform those to "\n" so we
  # have a single string
  string(REPLACE ";" "\n" code "${code}")
  set_property(TARGET rapids_export_${type}_${export_set} APPEND_STRING
               PROPERTY "${name}_POST_FIND_CODE" "${code}\n")
endfunction()
