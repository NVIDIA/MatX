# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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
                   [ENABLE_UNSTABLE]
                   [<CPM_ARGS> ...])

``ENABLE_UNSTABLE``
  Enable unstable features in CCCL.

.. |PKG_NAME| replace:: CCCL
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  CCCL::CCCL target will be created
  CCCL::Thrust target will be created
  CCCL::libcudacxx target will be created
  CCCL::CUB target will be created
  libcudacxx::libcudacxx target will be created
  CCCL::cudax target will be created (if ENABLE_UNSTABLE is specified)

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

  set(options ENABLE_UNSTABLE)
  set(one_value)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(CCCL ${_RAPIDS_UNPARSED_ARGUMENTS} VERSION_VAR version FIND_VAR find_args
                          CPM_VAR cpm_find_info TO_INSTALL_VAR to_install)

  if(_RAPIDS_ENABLE_UNSTABLE)
    list(APPEND cpm_find_info OPTIONS "CCCL_ENABLE_UNSTABLE ON")
  endif()

  if(to_install)
    # Make sure we install CCCL into the `include/rapids` subdirectory instead of the default
    include(GNUInstallDirs)
    string(APPEND CMAKE_INSTALL_INCLUDEDIR "/rapids")
    string(APPEND CMAKE_INSTALL_LIBDIR "/rapids")
    # We don't specify `CCCL_ENABLE_INSTALL_RULES` as a `rapids_cpm_find` argument so that it isn't
    # propagated to users of the build export file which would cause them to also install parts of
    # CCCL by mistake ( and in a wrong directory )
    get_property(rapids_cccl_install_rules_already_called GLOBAL
                 PROPERTY rapids_cmake_cccl_install_rules SET)
    if(NOT rapids_cccl_install_rules_already_called)
      set(CCCL_ENABLE_INSTALL_RULES ON)
      set(CUB_ENABLE_INSTALL_RULES ON)
      set(Thrust_ENABLE_INSTALL_RULES ON)
      set(libcudacxx_ENABLE_INSTALL_RULES ON)
      set_property(GLOBAL PROPERTY rapids_cmake_cccl_install_rules ON)
    endif()
  endif()

  # CCCL's install rules use ${CCCL_TOPLEVEL_PROJECT} as a positional argument. When CCCL is
  # included via add_subdirectory (not as the top-level project), this variable is undefined,
  # causing cmake_parse_arguments to misparse the function call. Explicitly set it to OFF.
  set(CCCL_TOPLEVEL_PROJECT OFF)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(CCCL ${version} ${find_args} GLOBAL_TARGETS CCCL CCCL::CCCL CCCL::CUB
                                                              CCCL::libcudacxx CCCL::cudax
                  CPM_ARGS FIND_PACKAGE_ARGUMENTS EXACT ${cpm_find_info})

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(CCCL)

  if(CCCL_SOURCE_DIR)
    # Store where CMake can find the cccl-config.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD CCCL "${CCCL_SOURCE_DIR}/lib/cmake/cccl"
                                    EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
    rapids_export_find_package_root(INSTALL CCCL
                                    [=[${CMAKE_CURRENT_LIST_DIR}/../../rapids/cmake/cccl]=]
                                    EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET} CONDITION to_install)
  endif()

  if(TARGET CCCL::CCCL)
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_DISABLE_NAMESPACE_MAGIC)
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_DISABLE_ABI_NAMESPACE)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE CCCL_DISABLE_PDL)
    set(post_find_code
        [=[
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_DISABLE_NAMESPACE_MAGIC)
    target_compile_definitions(CCCL::CCCL INTERFACE CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_DISABLE_ABI_NAMESPACE)
    target_compile_definitions(CCCL::CCCL INTERFACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    target_compile_definitions(CCCL::CCCL INTERFACE CCCL_DISABLE_PDL)
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
