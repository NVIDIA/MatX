# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cython_create_modules
----------------------------

.. versionadded:: v24.02.00

Generate C(++) from Cython and create Python modules.

.. code-block:: cmake

  rapids_cython_create_modules([CXX]
                               [SOURCE_FILES <src1> <src2> ...]
                               [LINKED_LIBRARIES <lib1> <lib2> ... ]
                               [INSTALL_DIR <install_path>]
                               [MODULE_PREFIX <module_prefix>]
                               [COMPONENT <component_name>]
                               [ASSOCIATED_TARGETS <target1> <target2> ...])

Creates a Cython target for each provided source file, then adds a
corresponding Python extension module. Each built library has its RPATH set to
$ORIGIN.

.. note::
  Requires :cmake:command:`rapids_cython_init` to be called before usage.

``CXX``
  Flag indicating that the Cython files need to generate C++ rather than C.

``SOURCE_FILES``
  The list of Cython source files to be built into Python extension modules.
  Note that this function assumes that Cython source files have a one-to-one
  correspondence with extension modules to build, i.e. for every `<Name>.pyx`
  in SOURCE_FILES we assume that `<Name>.pyx` is a Cython source file for which
  an extension module `<Name>` should be built.

``LINKED_LIBRARIES``
  The list of libraries that need to be linked into all modules. In RAPIDS,
  this list usually contains (at minimum) the corresponding C++ libraries.

``INSTALL_DIR``
  The path relative to the installation prefix so that it can be converted to
  an absolute path in a relocatable way. If not provided, defaults to the path
  to CMAKE_CURRENT_SOURCE_DIR relative to PROJECT_SOURCE_DIR.

``MODULE_PREFIX``
  A prefix used to name the generated library targets. This functionality is
  useful when multiple Cython modules in different subpackages of the the same
  project have the same name. The default prefix is the empty string.

``COMPONENT``
  The name of the component to which the generated Python modules should be
  installed. This allows for more granular control over the installation of
  different components of the project.

``ASSOCIATED_TARGETS``
  A list of targets that are associated with the Cython targets created in this
  function. The target to associated target mapping is stored and may be
  leveraged by the following functions:

  - :cmake:command:`rapids_cython_add_rpath_entries` accepts a path for an
    associated target and updates the RPATH of each target with which that
    associated target is associated.

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`RAPIDS_CYTHON_CREATED_TARGETS` will be set to a list of
  targets created by this function.

#]=======================================================================]

# cmake-lint: disable=R0915
function(rapids_cython_create_modules)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/verify_init.cmake)
  rapids_cython_verify_init()

  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cython.create_modules")

  set(_rapids_cython_options CXX)
  set(_rapids_cython_one_value INSTALL_DIR MODULE_PREFIX COMPONENT)
  set(_rapids_cython_multi_value SOURCE_FILES LINKED_LIBRARIES ASSOCIATED_TARGETS)
  cmake_parse_arguments(_RAPIDS_CYTHON "${_rapids_cython_options}" "${_rapids_cython_one_value}"
                        "${_rapids_cython_multi_value}" ${ARGN})

  set(_ext ".c")
  set(_language_flag "")
  if(_RAPIDS_CYTHON_CXX)
    set(_language_flag "--cplus")
    set(_ext ".cxx")
  endif()

  set(CREATED_TARGETS "")

  if(NOT DEFINED _RAPIDS_CYTHON_MODULE_PREFIX)
    set(_RAPIDS_CYTHON_MODULE_PREFIX "")
  endif()

  foreach(cython_filename IN LISTS _RAPIDS_CYTHON_SOURCE_FILES)
    # Generate a reasonable module name.
    cmake_path(GET cython_filename FILENAME cython_module)
    cmake_path(REPLACE_EXTENSION cython_module "${_ext}" OUTPUT_VARIABLE cpp_filename)
    cmake_path(REMOVE_EXTENSION cython_module)
    cmake_path(SET depfile NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/${cpp_filename}.dep")

    # Save the name of the module without the provided prefix so that we can control the output.
    set(cython_module_filename "${cython_module}")
    string(PREPEND cython_module ${_RAPIDS_CYTHON_MODULE_PREFIX})

    # Generate C++ from Cython and create a library for the resulting extension module to compile.
    # TODO: Probably want to generalize this to a helper function for invoking Cython.
    string(REPLACE " " ";" CYTHON_FLAGS_LIST "${CYTHON_FLAGS}")
    add_custom_command(OUTPUT ${cpp_filename}
                       DEPENDS ${cython_filename}
                       VERBATIM
                       COMMAND "${CYTHON}" ARGS ${_language_flag} -3 ${CYTHON_FLAGS_LIST}
                               "${CMAKE_CURRENT_SOURCE_DIR}/${cython_filename}" --output-file
                               "${CMAKE_CURRENT_BINARY_DIR}/${cpp_filename}" --depfile
                       DEPFILE ${depfile}
                       COMMENT "Transpiling ${cython_filename} to ${cpp_filename}")

    if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
      python_add_library(${cython_module} MODULE "${CMAKE_CURRENT_BINARY_DIR}/${cpp_filename}"
                         WITH_SOABI USE_SABI ${SKBUILD_SABI_VERSION})
    else()
      python_add_library(${cython_module} MODULE "${CMAKE_CURRENT_BINARY_DIR}/${cpp_filename}"
                         WITH_SOABI)
    endif()

    # The final library name must match the original filename and must ignore the prefix.
    set_target_properties(${cython_module} PROPERTIES LIBRARY_OUTPUT_NAME ${cython_module_filename})

    # Link the module to the requested libraries
    if(DEFINED _RAPIDS_CYTHON_LINKED_LIBRARIES)
      target_link_libraries(${cython_module} PUBLIC ${_RAPIDS_CYTHON_LINKED_LIBRARIES})
    endif()

    # Compute the install directory relative to the source and rely on installs being relative to
    # the CMAKE_PREFIX_PATH for e.g. editable installs.
    if(NOT DEFINED _RAPIDS_CYTHON_INSTALL_DIR)
      cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
                 OUTPUT_VARIABLE _RAPIDS_CYTHON_INSTALL_DIR)
    endif()
    set(component_arg)
    if(DEFINED _RAPIDS_CYTHON_COMPONENT)
      set(component_arg COMPONENT ${_RAPIDS_CYTHON_COMPONENT})
    endif()
    install(TARGETS ${cython_module} DESTINATION ${_RAPIDS_CYTHON_INSTALL_DIR} ${component_arg})

    # Default the INSTALL_RPATH for all modules to $ORIGIN.
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      set(platform_rpath_origin "@loader_path")
    else()
      set(platform_rpath_origin "$ORIGIN")
    endif()
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "${platform_rpath_origin}")

    # Store any provided associated targets in a global list
    foreach(associated_target IN LISTS _RAPIDS_CYTHON_ASSOCIATED_TARGETS)
      set_property(GLOBAL PROPERTY "rapids_cython_associations_${associated_target}"
                                   "${cython_module}" APPEND)
    endforeach()

    list(APPEND CREATED_TARGETS "${cython_module}")
  endforeach()

  set(RAPIDS_CYTHON_CREATED_TARGETS ${CREATED_TARGETS} PARENT_SCOPE)
endfunction()
