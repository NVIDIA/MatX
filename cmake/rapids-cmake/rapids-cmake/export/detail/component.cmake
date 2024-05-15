#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
rapids_export_component
-----------------------

.. versionadded:: v23.04.00

.. code-block:: cmake

Generate the necessary -Config.cmake modules needed for an optional component
of a rapids export package.

  rapids_export_component( (BUILD|INSTALL) <project_name> <component_name> <export_set> <unique_name> <namespace>)

The :cmake:command:`rapids_export_component` function generates
the `<proj>-<unique_name>-targets.cmake` and `<proj>-<unique_name>-dependencies.cmake`
files necessary for :cmake:command:`rapids_export` to support optional
components.

#]=======================================================================]
# cmake-lint: disable=W0105,R0913
function(rapids_export_component type project_name component_name export_set unique_name namespace)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.export.rapids_export_component")

  string(TOLOWER ${type} type)

  set(deps_destination)
  if(type STREQUAL "install")
    include("${rapids-cmake-dir}/cmake/install_lib_dir.cmake")
    rapids_cmake_install_lib_dir(install_location)
    set(install_location "${install_location}/cmake/${project_name}")

    set(deps_destination
        "${PROJECT_BINARY_DIR}/rapids-cmake/${project_name}/export/${component_name}/")
    file(MAKE_DIRECTORY "${deps_destination}")
    install(DIRECTORY "${deps_destination}" DESTINATION "${install_location}"
            COMPONENT ${component_name})
    if(namespace STREQUAL "")
      install(EXPORT ${export_set} FILE ${project_name}-${unique_name}-targets.cmake
              DESTINATION "${install_location}" COMPONENT ${component_name})
    else()
      install(EXPORT ${export_set}
              FILE ${project_name}-${unique_name}-targets.cmake
              DESTINATION "${install_location}"
              COMPONENT ${component_name}
              NAMESPACE ${namespace})
    endif()

  else()
    set(install_location "${PROJECT_BINARY_DIR}")
    set(deps_destination "${install_location}/")

    if(namespace STREQUAL "")
      export(EXPORT ${export_set}
             FILE "${install_location}/${project_name}-${unique_name}-targets.cmake")
    else()
      export(EXPORT ${export_set} NAMESPACE ${namespace}
             FILE "${install_location}/${project_name}-${unique_name}-targets.cmake")
    endif()

  endif()

  if(TARGET rapids_export_${type}_${export_set})
    include("${rapids-cmake-dir}/export/write_dependencies.cmake")
    rapids_export_write_dependencies(
      ${type} ${export_set} "${deps_destination}/${project_name}-${unique_name}-dependencies.cmake")
  endif()

endfunction()
