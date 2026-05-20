# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_export_find_package_file
-------------------------------

.. versionadded:: v21.06.00

Record that the file found at <file_path> needs to be usable as part of
the associated export set.

.. code-block:: cmake

  rapids_export_find_package_file( (BUILD|INSTALL)
                                   <file_path>
                                   EXPORT_SET <ExportSetName>
                                   [CONDITION <variableName>]
                                  )

When constructing export sets, espically installed ones it is
necessary to install any custom FindModules that your project
has written. The :cmake:command:`rapids_export_find_package_file(BUILD|INSTALL)`
allows projects to easily specify that a FindModule is either
used by our export set or used by a consumer, allowing
rapids-cmake to ensure it is installed correct and added to
:cmake:variable:`CMAKE_MODULE_PATH` when needed.

``BUILD``
  Record that the FindPackage at <file_path> needs to be part
  of our build directory export set. This means that it will be
  usable by the calling package if it needs to search for
  <PackageName> again.

``INSTALL``
  Record that the FindPackage at <file_path> needs to be part
  of our install export set. This means that it will be installed as
  part of our packages CMake export set infrastructure

``EXPORT_SET``
  List the export set name that this code should be attached too. If
  no name is given the associated call will be ignored.

``CONDITION``
  A boolean variable name, that when evaluates to undefined or a false value
  will cause the associated call to be ignored.

#]=======================================================================]
function(rapids_export_find_package_file type file_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.find_package_file")
  include("${rapids-cmake-dir}/cmake/detail/policy.cmake")

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

  # Don't remove duplicates here as that cost should only be paid once per export set. So that
  # should occur in `write_dependencies`

  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_PACKAGES_TO_INSTALL"
                                                                          "${file_path}")

endfunction()
