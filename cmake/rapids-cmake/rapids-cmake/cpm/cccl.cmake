#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
rapids_cpm_cccl
---------------

.. versionadded:: v24.02.00

Allow projects to find or build `CCCL` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of CCCL :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

When `BUILD_EXPORT_SET` is specified the generated build export set dependency
file will automatically call `thrust_create_target(CCCL::Thrust FROM_OPTIONS)`.

When `INSTALL_EXPORT_SET` is specified the generated install export set dependency
file will automatically call `thrust_create_target(CCCL::Thrust FROM_OPTIONS)`.

.. code-block:: cmake

  rapids_cpm_cccl( [BUILD_EXPORT_SET <export-name>]
                   [INSTALL_EXPORT_SET <export-name>]
                   [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: CCCL
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  CCCL::CCCL target will be created
  CCCL::Thrust target will be created
  CCCL::libcudacxx target will be created
  CCCL::CUB target will be created
  libcudacxx::libcudacxx target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`CCCL_SOURCE_DIR` is set to the path to the source directory of CCCL.
  :cmake:variable:`CCCL_BINARY_DIR` is set to the path to the build directory of CCCL.
  :cmake:variable:`CCCL_ADDED`      is set to a true value if CCCL has not been added before.
  :cmake:variable:`CCCL_VERSION`    is set to the version of CCCL specified by the versions.json.

#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_cpm_cccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.cccl")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(CCCL version repository tag shallow exclude)

  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN AND NOT exclude)
    set(to_install ON)
    # Make sure we install CCCL into the `include/rapids` subdirectory instead of the default
    include(GNUInstallDirs)
    set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/rapids")
    set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/rapids")
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(CCCL ${version} patch_command)

  # Ensure for CMake 3.24+ that the CCCL::Thrust target exists:
  # https://github.com/NVIDIA/cccl/pull/1182
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(CCCL ${version} ${ARGN}
                  GLOBAL_TARGETS CCCL CCCL::CCCL CCCL::CUB CCCL::libcudacxx
                  CPM_ARGS FIND_PACKAGE_ARGUMENTS EXACT
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${exclude}
                  OPTIONS "CCCL_ENABLE_INSTALL_RULES ${to_install}")

  # rapids_cpm_cccl can be called multiple times from the same scope such as from
  # cudf/CMakeLists.txt and cudf's call to find_package(rmm). In these situations, subsequent
  # invocations will still have `CCCL_SOURCE_DIR` set due to how `rapids_cpm_find` early termination
  # sets up variables
  #
  # So to properly preserve any custom install location values from the first invocation we need a
  # global property that we use to track that the cccl install rules have been called
  get_property(rapids_cccl_install_rules_already_called GLOBAL
               PROPERTY rapids_cmake_cccl_install_rules SET)
  if(CCCL_SOURCE_DIR AND to_install AND NOT rapids_cccl_install_rules_already_called)
    # CCCL does not currently correctly support installation of cub/thrust/libcudacxx. The only
    # option that makes this work is to manually invoke the install rules until CCCL's CMake is
    # fixed.
    set_property(GLOBAL PROPERTY rapids_cmake_cccl_install_rules ON)
    set(Thrust_SOURCE_DIR "${CCCL_SOURCE_DIR}/thrust")
    set(CUB_SOURCE_DIR "${CCCL_SOURCE_DIR}/cub")
    set(libcudacxx_SOURCE_DIR "${CCCL_SOURCE_DIR}/libcudacxx")

    set(Thrust_BINARY_DIR "${CCCL_BINARY_DIR}")
    set(CUB_BINARY_DIR "${CCCL_BINARY_DIR}")
    set(libcudacxx_BINARY_DIR "${CCCL_BINARY_DIR}")

    set(Thrust_ENABLE_INSTALL_RULES ON)
    set(CUB_ENABLE_INSTALL_RULES ON)
    set(libcudacxx_ENABLE_INSTALL_RULES ON)

    include("${Thrust_SOURCE_DIR}/cmake/ThrustInstallRules.cmake")
    include("${CUB_SOURCE_DIR}/cmake/CubInstallRules.cmake")

    # libcudacxx's install rules require inserting an extra level of nesting for the include dir.
    set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/libcudacxx")
    include("${libcudacxx_SOURCE_DIR}/cmake/libcudacxxInstallRules.cmake")
  endif()

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(CCCL)

  set(options)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(CCCL_SOURCE_DIR)
    # Store where CMake can find the Thrust-config.cmake that comes part of Thrust source code
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD CCCL "${CCCL_SOURCE_DIR}/cmake"
                                    EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
    rapids_export_find_package_root(INSTALL CCCL
                                    [=[${CMAKE_CURRENT_LIST_DIR}/../../rapids/cmake/cccl]=]
                                    EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET} CONDITION to_install)
  endif()

  if(TARGET CCCL::CCCL)
    # Can be removed once we move to CCCL 2.3
    #
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_DISABLE_ABI_NAMESPACE)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    set(post_find_code
        [=[
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_DISABLE_ABI_NAMESPACE)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    ]=])
    include("${rapids-cmake-dir}/export/detail/post_find_package_code.cmake")
    rapids_export_post_find_package_code(BUILD CCCL "${post_find_code}" EXPORT_SET
                                         ${_RAPIDS_BUILD_EXPORT_SET})
    rapids_export_post_find_package_code(INSTALL CCCL "${post_find_code}" EXPORT_SET
                                         ${_RAPIDS_INSTALL_EXPORT_SET} CONDITION to_install)
  endif()

  # Propagate up variables that CPMFindPackage provides
  set(CCCL_SOURCE_DIR "${CCCL_SOURCE_DIR}" PARENT_SCOPE)
  set(CCCL_BINARY_DIR "${CCCL_BINARY_DIR}" PARENT_SCOPE)
  set(CCCL_ADDED "${CCCL_ADDED}" PARENT_SCOPE)
  set(CCCL_VERSION ${version} PARENT_SCOPE)

endfunction()
