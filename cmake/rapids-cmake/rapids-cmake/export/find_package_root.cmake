#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
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
                                   <ExportSet>
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

#]=======================================================================]
function(rapids_export_find_package_root type name dir_path export_set)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.find_package_root_dir")

  string(TOLOWER ${type} type)

  if(NOT TARGET rapids_export_${type}_${export_set})
    add_library(rapids_export_${type}_${export_set} INTERFACE)
  endif()

  # Don't remove duplicates here as that cost should only be paid Once per export set. So that
  # should occur in `write_dependencies`
  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_ROOT_PACKAGES"
                                                                          ${name})
  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_ROOT_FOR_${name}"
                                                                          ${dir_path})

endfunction()
