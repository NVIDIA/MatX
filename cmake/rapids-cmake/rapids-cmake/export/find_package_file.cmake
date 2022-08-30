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
rapids_export_find_package_file
-------------------------------

.. versionadded:: v21.06.00

Record that the file found at <file_path> needs to be usable as part of
the associated export set.

.. code-block:: cmake

  rapids_export_find_package_file( (BUILD|INSTALL)
                                   <file_path>
                                   <ExportSet>
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

#]=======================================================================]
function(rapids_export_find_package_file type file_path export_set)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.find_package_file")

  string(TOLOWER ${type} type)

  if(NOT TARGET rapids_export_${type}_${export_set})
    add_library(rapids_export_${type}_${export_set} INTERFACE)
  endif()

  # Don't remove duplicates here as that cost should only be paid Once per export set. So that
  # should occur in `write_dependencies`

  set_property(TARGET rapids_export_${type}_${export_set} APPEND PROPERTY "FIND_PACKAGES_TO_INSTALL"
                                                                          "${file_path}")

endfunction()
