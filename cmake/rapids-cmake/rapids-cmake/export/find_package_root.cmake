# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_export_find_package_root
-------------------------------

.. versionadded:: v21.10.00

Record that for <PackageName> to be found correctly, the :cmake:variable:`<PackageName>_ROOT_DIR`
needs to be set to the provided path.

.. code-block:: cmake

  rapids_export_find_package_root( (BUILD|INSTALL)
                                   <PackageName>
                                   <directory_path>
                                   EXPORT_SET <ExportSetName>
                                   [CONDITION <variableName>]
                                   )

When constructing complicated export sets, espically ones that
install complicated dependencies it can be necessary to specify
:cmake:variable:`PackageName_ROOT` so that we are sure we
will find the packaged dependency.

``BUILD``
  Record that the `PackageName_ROOT` will be set to <directory_path>
  before any find_dependency calls for `PackageName` for our build directory
  export set.

``INSTALL``
  Record that the `PackageName_ROOT` will be set to <directory_path>
  before any find_dependency calls for `PackageName` for our install directory
  export set.

``EXPORT_SET``
  List the export set name that the `directory_path` should be attached too. If
  no name is given the associated call will be ignored.

``CONDITION``
  A boolean variable name, that when evaluates to undefined or a false value
  will cause the associated call to be ignored.

#]=======================================================================]
function(rapids_export_find_package_root type name dir_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.find_package_root_dir")
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
  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_ROOT_PACKAGES"
                                                                          ${name})
  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_ROOT_FOR_${name}"
                                                                          ${dir_path})

endfunction()
