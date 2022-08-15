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
rapids_cpm_thrust
-----------------

.. versionadded:: v21.10.00

Allow projects to find or build `Thrust` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of Thrust :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_thrust( NAMESPACE <namespace>
                     [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [<CPM_ARGS> ...])

``NAMESPACE``
  The namespace that the Thrust target will be constructed into.

.. |PKG_NAME| replace:: Thrust
.. include:: common_package_args.txt

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
function(rapids_cpm_thrust NAMESPACE namespaces_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.thrust")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Thrust version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(Thrust ${version} ${ARGN}
                  GLOBAL_TARGETS ${namespaces_name}::Thrust
                  CPM_ARGS FIND_PACKAGE_ARGUMENTS EXACT
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "THRUST_ENABLE_INSTALL_RULES OFF")

  if(NOT TARGET ${namespaces_name}::Thrust)
    thrust_create_target(${namespaces_name}::Thrust FROM_OPTIONS)
  endif()

  # Since `GLOBAL_TARGET ${namespaces_name}::Thrust` will list the target to be promoted to global
  # by `rapids_export` this will break consumers as the target doesn't exist when generating the
  # dependencies.cmake file, but requires a call to `thrust_create_target`
  #
  # So determine what `BUILD_EXPORT_SET` and `INSTALL_EXPORT_SET` this was added to and remove
  # ${namespaces_name}::Thrust
  set(options CPM_ARGS)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(_RAPIDS_BUILD_EXPORT_SET)
    set(target_name rapids_export_build_${_RAPIDS_BUILD_EXPORT_SET})
    get_target_property(global_targets ${target_name} GLOBAL_TARGETS)
    list(REMOVE_ITEM global_targets "${namespaces_name}::Thrust")
    set_target_properties(${target_name} PROPERTIES GLOBAL_TARGETS "${global_targets}")
  endif()

  if(_RAPIDS_INSTALL_EXPORT_SET)
    set(target_name rapids_export_install_${_RAPIDS_INSTALL_EXPORT_SET})
    get_target_property(global_targets ${target_name} GLOBAL_TARGETS)
    list(REMOVE_ITEM global_targets "${namespaces_name}::Thrust")
    set_target_properties(${target_name} PROPERTIES GLOBAL_TARGETS "${global_targets}")
  endif()

  # only install thrust when we have an in-source version
  if(Thrust_SOURCE_DIR AND _RAPIDS_INSTALL_EXPORT_SET)
    #[==[
    Projects such as cudf, and rmm require a newer versions of thrust than can be found in the oldest supported CUDA toolkit.
    This requires these components to install/packaged so that consumers use the same version. To make sure that the custom
    version of thrust is used over the CUDA toolkit version we need to ensure we always use an user include and not a system.

    By default if we allow thrust to install into `CMAKE_INSTALL_INCLUDEDIR` alongside rmm (or other pacakges)
    we will get a install tree that looks like this:

      install/include/rmm
      install/include/cub
      install/include/thrust

    This is a problem for CMake+NVCC due to the rules around import targets, and user/system includes. In this case both
    rmm and thrust will specify an include path of `install/include`, while thrust tries to mark it as an user include,
    since rmm uses CMake's default of system include. Compilers when provided the same include as both user and system
    always goes with system.

    Now while rmm could also mark `install/include` as system this just pushes the issue to another dependency which
    isn't built by RAPIDS and comes by and marks `install/include` as system.

    Instead the more reliable option is to make sure that we get thrust to be placed in an unique include path that
    to other project will use. In the case of rapids-cmake we install the headers to `include/rapids/thrust`
    #]==]
    include(GNUInstallDirs)
    install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/thrust/" FILES_MATCHING
            REGEX "\\.(h|inl)$")
    install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/thrust/dependencies/" FILES_MATCHING
            PATTERN "*.cuh")

    install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/thrust/thrust/")
    install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub/cmake"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/thrust/dependencies/cub/")

    include("${rapids-cmake-dir}/cmake/install_lib_dir.cmake")
    rapids_cmake_install_lib_dir(install_location) # Use the correct conda aware path

    # We need to install the forwarders in `lib/cmake/thrust` and `lib/cmake/cub`
    set(scratch_dir
        "${CMAKE_BINARY_DIR}/rapids-cmake/${_RAPIDS_INSTALL_EXPORT_SET}/install/scratch/")

    file(WRITE "${scratch_dir}/thrust-config.cmake"
         [=[include("${CMAKE_CURRENT_LIST_DIR}/../../../include/rapids/thrust/thrust/cmake/thrust-config.cmake")]=]
    )
    file(WRITE "${scratch_dir}/thrust-config-version.cmake"
         [=[include("${CMAKE_CURRENT_LIST_DIR}/../../../include/rapids/thrust/thrust/cmake/thrust-config-version.cmake")]=]
    )
    install(FILES "${scratch_dir}/thrust-config.cmake" "${scratch_dir}/thrust-config-version.cmake"
            DESTINATION "${install_location}/cmake/thrust")

    file(WRITE "${scratch_dir}/cub-config.cmake"
         [=[include("${CMAKE_CURRENT_LIST_DIR}/../../../include/rapids/thrust/dependencies/cub/cub-config.cmake")]=]
    )
    file(WRITE "${scratch_dir}/cub-config-version.cmake"
         [=[include("${CMAKE_CURRENT_LIST_DIR}/../../../include/rapids/thrust/dependencies/cub/cub-config-version.cmake")]=]
    )
    install(FILES "${scratch_dir}/cub-config.cmake" "${scratch_dir}/cub-config-version.cmake"
            DESTINATION "${install_location}/cmake/cub")
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(Thrust_SOURCE_DIR "${Thrust_SOURCE_DIR}" PARENT_SCOPE)
  set(Thrust_BINARY_DIR "${Thrust_BINARY_DIR}" PARENT_SCOPE)
  set(Thrust_ADDED "${Thrust_ADDED}" PARENT_SCOPE)
  set(Thrust_VERSION ${version} PARENT_SCOPE)

endfunction()
