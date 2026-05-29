# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_find
---------------

.. versionadded:: v21.06.00

Allow projects to find or build arbitrary projects via `CPM` with built-in
tracking of these dependencies for correct export support.

.. code-block:: cmake

  rapids_cpm_find(<PackageName> <version>
                  [COMPONENTS <components...>]
                  [GLOBAL_TARGETS <targets...>]
                  [BUILD_EXPORT_SET <export-name>]
                  [INSTALL_EXPORT_SET <export-name>]
                  <CPM_ARGS>
                    all normal CPM options
                )

Generate a CPM FindPackage call and associate this with the listed
build and install export set for correct export generation.

Since the visibility of CMake's targets differ between targets built locally and those
imported, :cmake:command:`rapids_cpm_find` promotes imported targets to be global so users have
consistency. List all targets used by your project in `GLOBAL_TARGET`.

.. note::
  Requires :cmake:command:`rapids_cpm_init` to be called before usage

``PackageName``
  Name of the package to find.

``version``
  Version of the package you would like CPM to find.

``COMPONENTS``
  .. versionadded:: v22.10.00

  A list of required components that are required to be found for this
  package to be considered valid when doing a local search.

``GLOBAL_TARGETS``
  Which targets from this package should be made global. This information
  will be propagated to any associated export set.

  .. versionchanged:: v21.10.00
    If any targets listed in `GLOBAL_TARGET` exist when :cmake:command:`rapids_cpm_find` is called
    no calls to `CPM` will be executed. This is done for the following reasons:

      - Removes the need for the calling code to do the conditional checks
      - Allows `BUILD_EXPORT_SET` and `INSTALL_EXPORT_SET` tracking to happen correctly when targets had already been brought it by non-CPM means.

``BUILD_EXPORT_SET``
  Record that a :cmake:command:`CPMFindPackage(<PackageName> ...)` call needs to occur as part of
  our build directory export set.

``INSTALL_EXPORT_SET``
  Record a :cmake:command:`find_dependency(<PackageName> ...) <cmake:module:CMakeFindDependencyMacro>` call needs to occur as part of
  our install directory export set.

``BUILD_PATCH_ONLY``
  Do not require the package to be downloaded even if a patch command is present. This is to enable patches that only affect
  the build process and not runtime functionality.

``CPM_ARGS``
  Required placeholder to be provided before any extra arguments that need to
  be passed down to :cmake:command:`CPMFindPackage`.

  .. note::
    A ``PATCH_COMMAND`` will always trigger usage of :cmake:command:`CPMAddPackage` instead of :cmake:command:`CPMFindPackage`,
    unless ``BUILD_PATCH_ONLY`` is specified. *This is true even if the patch command is empty.*

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`<PackageName>_SOURCE_DIR` is set to the path to the source directory of <PackageName>.
  :cmake:variable:`<PackageName>_BINARY_DIR`  is set to the path to the build directory of  <PackageName>.
  :cmake:variable:`<PackageName>_ADDED`      is set to a true value if <PackageName> has not been added before.

.. note::
  Adding an export set to :cmake:command:`rapids_cpm_find` has different behavior
  for build and install. Build exports a respective CPM call, since
  we presume other CPM packages don't generate a correct build directory
  config module. While install exports a `find_dependency` call as
  we expect projects to have a valid install setup.

  If you need different behavior you will need to use :cmake:command:`rapids_export_package()`
  or :cmake:command:`rapids_export_cpm()`.

  If :cmake:variable:`CPM_<PackageName>_SOURCE` is set, we use :cmake:command:`CPMAddPackage` instead of
  :cmake:command:`CPMFindPackage`. :cmake:command:`CPMAddPackage` always adds the package at the desired
  :cmake:variable:`CPM_<PackageName>_SOURCE` location, and won't attempt to locate it via
  :cmake:command:`find_package() <cmake:command:find_package>` first.


Examples
^^^^^^^^

Example on how to use :cmake:command:`rapids_cpm_find` to include common projects


.. code-block:: cmake

  # fmt
  rapids_cpm_find(fmt 8.0.1
    GLOBAL_TARGETS fmt::fmt
    CPM_ARGS
      GITHUB_REPOSITORY fmtlib/fmt
      GIT_TAG 8.0.1
      GIT_SHALLOW TRUE
  )

  # google benchmark, no GIT_TAG required since it uses `v<Version>` tags
  rapids_cpm_find(benchmark 1.5.2
    CPM_ARGS
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_SHALLOW     TRUE
        OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                        "BENCHMARK_ENABLE_INSTALL OFF"
  )

Overriding
^^^^^^^^^^

The :cmake:command:`rapids_cpm_package_override` command provides a way
for projects to override the default values for any :cmake:command:`rapids_cpm_find`, :ref:`rapids_cpm_* <cpm_pre-configured_packages>`,
`CPM <https://github.com/cpm-cmake/CPM.cmake>`_, and :cmake:module:`FetchContent() <cmake:module:FetchContent>` package.

By default when an override for a project is provided no local search
for that project will occur. This is done to make sure that the requested
modified version is used.

#]=======================================================================]
# cmake-lint: disable=R0912,R0915
function(rapids_cpm_find name version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.find")
  set(options CPM_ARGS BUILD_PATCH_ONLY)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET SOURCE_SUBDIR)
  set(multi_value COMPONENTS GLOBAL_TARGETS)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _RAPIDS_CPM_ARGS)
    message(FATAL_ERROR "rapids_cpm_find requires you to specify CPM_ARGS before any CPM arguments")
  endif()

  set(has_non_build_patch FALSE)
  if(NOT _RAPIDS_BUILD_PATCH_ONLY)
    foreach(unparsed_arg IN LISTS _RAPIDS_UNPARSED_ARGUMENTS)
      if(unparsed_arg STREQUAL "PATCH_COMMAND")
        set(has_non_build_patch TRUE)
        break()
      endif()
    endforeach()
  endif()

  set(package_needs_to_be_added TRUE)
  if(_RAPIDS_GLOBAL_TARGETS)
    foreach(target IN LISTS _RAPIDS_GLOBAL_TARGETS)
      if(TARGET ${target})
        set(package_needs_to_be_added FALSE)
        break()
      endif()
    endforeach()
  endif()

  if(_RAPIDS_COMPONENTS)
    # We need to pass the set of components as a space separated string and not a list
    string(REPLACE ";" " " _RAPIDS_COMPONENTS "${_RAPIDS_COMPONENTS}")
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS "FIND_PACKAGE_ARGUMENTS"
         "COMPONENTS ${_RAPIDS_COMPONENTS}")
  endif()

  if(_RAPIDS_SOURCE_SUBDIR)
    # We have seen issues with some versions of CMake ( 4.0.Y ) not properly recording the
    # SOURCE_SUBDIR in the stored information return by FetchContent_GetProperties
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS "SOURCE_SUBDIR" "${_RAPIDS_SOURCE_SUBDIR}")
    # Store this in a property that `pinning_write_file` can query
    set_property(GLOBAL PROPERTY rapids_cmake_${name}_SOURCE_SUBDIR "${_RAPIDS_SOURCE_SUBDIR}")
  endif()

  if(package_needs_to_be_added)
    # Any non-build patch command triggers CPMAddPackage.
    if(CPM_${name}_SOURCE OR has_non_build_patch)
      message(DEBUG
              "rapids.cpm.rapids_find: CPMAddPackage(NAME ${name} VERSION ${version} ${_RAPIDS_UNPARSED_ARGUMENTS})"
      )
      CPMAddPackage(NAME ${name} VERSION ${version} ${_RAPIDS_UNPARSED_ARGUMENTS})
    else()
      message(DEBUG
              "rapids.cpm.rapids_find CPMFindPackage(NAME ${name} VERSION ${version} ${_RAPIDS_UNPARSED_ARGUMENTS})"
      )
      CPMFindPackage(NAME ${name} VERSION ${version} ${_RAPIDS_UNPARSED_ARGUMENTS})
    endif()
  else()
    # Restore any CPM variables that might be cached
    cpm_check_if_package_already_added(${name} ${version})
    if(CPM_PACKAGE_ALREADY_ADDED)
      cpm_export_variables(${name})
    endif()
  endif()

  set(_rapids_extra_info)
  if(_RAPIDS_GLOBAL_TARGETS)
    include("${rapids-cmake-dir}/cmake/make_global.cmake")
    rapids_cmake_make_global(_RAPIDS_GLOBAL_TARGETS)

    list(APPEND _rapids_extra_info "GLOBAL_TARGETS" ${_RAPIDS_GLOBAL_TARGETS})
  endif()

  if(_RAPIDS_BUILD_EXPORT_SET)
    include("${rapids-cmake-dir}/export/cpm.cmake")
    rapids_export_cpm(BUILD ${name} ${_RAPIDS_BUILD_EXPORT_SET}
                      CPM_ARGS NAME ${name} VERSION ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                               ${_rapids_extra_info})
  endif()

  if(_RAPIDS_INSTALL_EXPORT_SET)
    include("${rapids-cmake-dir}/export/package.cmake")

    if(_RAPIDS_COMPONENTS)
      list(APPEND _rapids_extra_info "COMPONENTS" ${_RAPIDS_COMPONENTS})
    endif()
    rapids_export_package(INSTALL ${name} ${_RAPIDS_INSTALL_EXPORT_SET} VERSION ${version}
                                                                        ${_rapids_extra_info})
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(${name}_SOURCE_DIR "${${name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${name}_BINARY_DIR "${${name}_BINARY_DIR}" PARENT_SCOPE)
  set(${name}_ADDED "${${name}_ADDED}" PARENT_SCOPE)

endfunction()
