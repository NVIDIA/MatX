# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_export_package
---------------------

.. versionadded:: v21.06.00

Record a given <PackageName> found by :cmake:command:`find_package <cmake:command:find_package>`
is required for a given export set

.. code-block:: cmake

  rapids_export_package( (BUILD|INSTALL)
                         <PackageName>
                         <ExportSet>
                         [VERSION] major.minor
                         [GLOBAL_TARGETS <targets...>]
                        )

Records a given <PackageName> found by :cmake:command:`find_package <cmake:command:find_package>`
is required for a given export set. When the associated :cmake:command:`rapids_export(BUILD|INSTALL)` or
:cmake:command:`rapids_export_write_dependencies(BUILD|INSTALL)` command is invoked the
generated information will include a :cmake:command:`find_dependency` call for <PackageName>.

``BUILD``
  Will record <PackageName> is part of the build directory export set

``INSTALL``
  Will record <PackageName> is part of the install directory export set

``VERSION``
  .. versionadded:: v22.04.00

  Record which `major.minor` version of the package is required for consumers.

``COMPONENTS``
  .. versionadded:: v22.10.00

  Record which components of the package are required for consumers.

``GLOBAL_TARGETS``
  Which targets from this package should be made global when the
  package is imported in.


#]=======================================================================]
function(rapids_export_package type name export_set)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.package")

  string(TOLOWER ${type} type)

  set(options "")
  set(one_value EXPORT_SET VERSION)
  set(multi_value GLOBAL_TARGETS COMPONENTS)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(type STREQUAL build)
    if(DEFINED ${name}_DIR AND ${name}_DIR)
      # Export out where we found the existing local config module
      set(possible_dir "${${name}_DIR}")
    endif()
  endif()

  if(_RAPIDS_COMPONENTS AND _RAPIDS_VERSION)
    set(version ${_RAPIDS_VERSION})
    set(components ${_RAPIDS_COMPONENTS})
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/${type}_package_components_versioned.cmake.in"
                   "${CMAKE_BINARY_DIR}/rapids-cmake/${export_set}/${type}/package_${name}.cmake"
                   @ONLY)
  elseif(_RAPIDS_COMPONENTS)
    set(components ${_RAPIDS_COMPONENTS})
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/${type}_package_components.cmake.in"
                   "${CMAKE_BINARY_DIR}/rapids-cmake/${export_set}/${type}/package_${name}.cmake"
                   @ONLY)
  elseif(_RAPIDS_VERSION)
    set(version ${_RAPIDS_VERSION})
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/${type}_package_versioned.cmake.in"
                   "${CMAKE_BINARY_DIR}/rapids-cmake/${export_set}/${type}/package_${name}.cmake"
                   @ONLY)
  else()
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/${type}_package.cmake.in"
                   "${CMAKE_BINARY_DIR}/rapids-cmake/${export_set}/${type}/package_${name}.cmake"
                   @ONLY)
  endif()

  if(NOT TARGET rapids_export_${type}_${export_set})
    add_library(rapids_export_${type}_${export_set} INTERFACE)
  endif()

  # Don't remove duplicates here as that cost should only be paid once per export set. So that
  # should occur in `write_dependencies`

  # Need to record the <PackageName> to `rapids_export_${type}_${export_set}`
  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "PACKAGE_NAMES" "${name}")

  if(_RAPIDS_GLOBAL_TARGETS)
    # record our targets that need to be marked as global when imported
    set_property(TARGET rapids_export_${type}_${export_set} APPEND
                 PROPERTY "GLOBAL_TARGETS" "${_RAPIDS_GLOBAL_TARGETS}")
  endif()

endfunction()
