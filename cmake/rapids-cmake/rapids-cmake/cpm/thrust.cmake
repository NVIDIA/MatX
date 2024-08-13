#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
rapids_cpm_thrust
-----------------

.. versionadded:: v21.10.00

Allow projects to find or build `Thrust` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of Thrust :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. deprecated:: v24.04.00
  ``rapids_cpm_thrust`` uses Thrust 1.x. Users should migrate to
  ``rapids_cpm_cccl`` which uses CCCL 2.x, including new versions of Thrust,
  CUB, and libcudacxx.

.. code-block:: cmake

  rapids_cpm_thrust( NAMESPACE <namespace>
                     [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [<CPM_ARGS> ...])

``NAMESPACE``
  The namespace that the Thrust target will be constructed into.

.. |PKG_NAME| replace:: Thrust
.. include:: common_package_args.txt

.. versionadded:: v23.12.00
  When `BUILD_EXPORT_SET` is specified the generated build export set dependency
  file will automatically call `thrust_create_target(<namespace>::Thrust FROM_OPTIONS)`.

  When `INSTALL_EXPORT_SET` is specified the generated install export set dependency
  file will automatically call `thrust_create_target(<namespace>::Thrust FROM_OPTIONS)`.

Result Targets
^^^^^^^^^^^^^^
  <namespace>::Thrust target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`Thrust_SOURCE_DIR` is set to the path to the source directory of Thrust.
  :cmake:variable:`Thrust_BINARY_DIR` is set to the path to the build directory of  Thrust.
  :cmake:variable:`Thrust_ADDED`      is set to a true value if Thrust has not been added before.
  :cmake:variable:`Thrust_VERSION`    is set to the version of Thrust specified by the versions.json.

#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_cpm_thrust NAMESPACE namespaces_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.thrust")

  include("${rapids-cmake-dir}/cmake/detail/policy.cmake")
  rapids_cmake_policy(DEPRECATED_IN 24.04
                      REMOVED_IN 24.10
                      MESSAGE [=[Usage of `rapids_cpm_thrust` has been deprecated in favor of `rapids_cpm_cccl`.]=]
  )

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Thrust version repository tag shallow exclude)

  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN AND NOT exclude)
    set(to_install ON)
    # Make sure we install thrust into the `include/rapids` subdirectory instead of the default
    include(GNUInstallDirs)
    set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/rapids")
    set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/rapids")
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(Thrust ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(Thrust ${version} ${ARGN}
                  GLOBAL_TARGETS ${namespaces_name}::Thrust
                  CPM_ARGS FIND_PACKAGE_ARGUMENTS EXACT
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "THRUST_ENABLE_INSTALL_RULES ${to_install}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(Thrust)

  set(options)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  set(post_find_code "if(NOT TARGET ${namespaces_name}::Thrust)"
                     "  thrust_create_target(${namespaces_name}::Thrust FROM_OPTIONS)" "endif()")

  if(Thrust_SOURCE_DIR)
    # Store where CMake can find the Thrust-config.cmake that comes part of Thrust source code
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    include("${rapids-cmake-dir}/export/detail/post_find_package_code.cmake")
    rapids_export_find_package_root(BUILD Thrust "${Thrust_SOURCE_DIR}/cmake"
                                    EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
    rapids_export_post_find_package_code(BUILD Thrust "${post_find_code}" EXPORT_SET
                                         ${_RAPIDS_BUILD_EXPORT_SET})

    rapids_export_find_package_root(INSTALL Thrust
                                    [=[${CMAKE_CURRENT_LIST_DIR}/../../rapids/cmake/thrust]=]
                                    EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET} CONDITION to_install)
    rapids_export_post_find_package_code(INSTALL Thrust "${post_find_code}" EXPORT_SET
                                         ${_RAPIDS_INSTALL_EXPORT_SET} CONDITION to_install)
  endif()

  # Check for the existence of thrust_create_target so we support fetching Thrust with DOWNLOAD_ONLY
  if(NOT TARGET ${namespaces_name}::Thrust AND COMMAND thrust_create_target)
    thrust_create_target(${namespaces_name}::Thrust FROM_OPTIONS)
    set_target_properties(${namespaces_name}::Thrust PROPERTIES IMPORTED_NO_SYSTEM ON)
    if(TARGET _Thrust_Thrust)
      set_target_properties(_Thrust_Thrust PROPERTIES IMPORTED_NO_SYSTEM ON)
    endif()
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(Thrust_SOURCE_DIR "${Thrust_SOURCE_DIR}" PARENT_SCOPE)
  set(Thrust_BINARY_DIR "${Thrust_BINARY_DIR}" PARENT_SCOPE)
  set(Thrust_ADDED "${Thrust_ADDED}" PARENT_SCOPE)
  set(Thrust_VERSION ${version} PARENT_SCOPE)

endfunction()
