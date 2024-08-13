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

cmake_policy(PUSH)
cmake_policy(VERSION 3.23)

#[=======================================================================[.rst:
rapids_find_generate_module
---------------------------

.. versionadded:: v21.06.00

Generate a Find*.cmake module for the requested package

.. code-block:: cmake

  rapids_find_generate_module( <PackageName>
                  HEADER_NAMES <paths...>
                  [LIBRARY_NAMES <names...>]
                  [INCLUDE_SUFFIXES <suffixes...>]
                  [VERSION <version>]
                  [NO_CONFIG]
                  [INITIAL_CODE_BLOCK <code_block_variable>]
                  [FINAL_CODE_BLOCK <code_block_variable>]
                  [BUILD_EXPORT_SET <name>]
                  [INSTALL_EXPORT_SET <name>]
                  )

Generates a custom Find module for the requested package. Makes
it easier for projects to look for packages that don't have
an existing FindModule or don't provide a CONFIG module
when installed.

.. note::
  If you are using this for a module that is part of
  your BUILD or INSTALL export set, it is highly likely
  that this needs to be part of the same export sets.


``HEADER_NAMES``
  Header names that should be provided to :cmake:command:`find_path` to
  determine the include directory of the package. If provided
  a list of names only one needs to be found for a directory
  to be considered a match

``LIBRARY_NAMES``
  library names that should be provided to :cmake:command:`find_library` to
  determine the include directory of the package. If provided
  a list of names only one needs to be found for a directory
  to be considered a match

  .. note::
    Every entry that doesn't start with `lib` will also be
    searched for as `lib<name>`

``INCLUDE_SUFFIXES``
  Extra relative sub-directories to use while searching for `HEADER_NAMES`.

``VERSION``
  Will append extra entries of the library to search for based on the
  content of `LIBRARY_NAMES`:

    - <name><version>
    - <name>.<version>
    - lib<name><version>
    - lib<name>.<version>

  This ordering is done explicitly to follow CMake recommendations
  for searching for versioned libraries:

    "We recommend specifying the unversioned name first so that locally-built packages
    can be found before those provided by distributions."

``NO_CONFIG``
  When provided will stop the generated Find Module from
  first searching for the projects shipped Find Config.

``INITIAL_CODE_BLOCK``
  Optional value of the variable that holds a string of code that will
  be executed as the first step of this config file.

  Note: This requires the code block variable instead of the contents
  so that we can properly insert CMake code

``FINAL_CODE_BLOCK``
  Optional value of the variable that holds a string of code that will
  be executed as the last step of this config file.

  Note: This requires the code block variable instead of the contents
  so that we can properly insert CMake code

``BUILD_EXPORT_SET``
  Record that this custom FindPackage module needs to be part
  of our build directory export set. This means that it will be
  usable by the calling package if it needs to search for
  <PackageName> again.

``INSTALL_EXPORT_SET``
  Record that this custom FindPackage module needs to be part
  of our install export set. This means that it will be installed as
  part of our packages CMake export set infrastructure

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`CMAKE_MODULE_PATH` will be modified to include the
  folder where `Find<PackageName>.cmake` is located.


Example on how to properly use :cmake:command:`rapids_find_generate_module`:

.. code-block:: cmake

  ...

  rapids_find_generate_module(
    RDKAFKA
    HEADER_NAMES rdkafkacpp.h
    LIBRARY_NAMES rdkafka++
    BUILD_EXPORT_SET consumer-exports
    INSTALL_EXPORT_SET consumer-exports
  )
  rapids_find_package(
    RDKAFKA REQUIRED
    BUILD_EXPORT_SET consumer-exports
    INSTALL_EXPORT_SET consumer-exports
  )


#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_find_generate_module name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.find.generate_module")

  set(options NO_CONFIG)
  set(one_value VERSION BUILD_EXPORT_SET INSTALL_EXPORT_SET INITIAL_CODE_BLOCK FINAL_CODE_BLOCK)
  set(multi_value HEADER_NAMES LIBRARY_NAMES INCLUDE_SUFFIXES)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _RAPIDS_HEADER_NAMES)
    message(FATAL_ERROR "rapids_find_generate_module requires HEADER_NAMES to be provided")
  endif()

  set(_RAPIDS_PKG_NAME ${name})

  # Construct any extra suffix search paths
  set(_RAPIDS_PATH_SEARCH_ARGS)
  if(_RAPIDS_INCLUDE_SUFFIXES)
    string(APPEND _RAPIDS_PATH_SEARCH_ARGS "PATH_SUFFIXES ${_RAPIDS_INCLUDE_SUFFIXES}")
  endif()

  set(_RAPIDS_HEADER_ONLY TRUE)
  if(DEFINED _RAPIDS_LIBRARY_NAMES)
    set(_RAPIDS_HEADER_ONLY FALSE)

    # Construct the release and debug library names handling version number suffixes
    set(_RAPIDS_PKG_LIB_NAMES ${_RAPIDS_LIBRARY_NAMES})
    set(_RAPIDS_PKG_LIB_DEBUG_NAMES ${_RAPIDS_LIBRARY_NAMES})
    list(TRANSFORM _RAPIDS_PKG_LIB_DEBUG_NAMES APPEND "d")

    if(DEFINED _RAPIDS_VERSION)
      list(TRANSFORM _RAPIDS_PKG_LIB_NAMES APPEND "${_RAPIDS_VERSION}" OUTPUT_VARIABLE lib_version1)
      list(TRANSFORM _RAPIDS_PKG_LIB_NAMES APPEND ".${_RAPIDS_VERSION}" OUTPUT_VARIABLE
                                                                        lib_version2)
      list(PREPEND _RAPIDS_PKG_LIB_NAMES ${lib_version1} ${lib_version2})

      list(TRANSFORM _RAPIDS_PKG_LIB_DEBUG_NAMES APPEND "${_RAPIDS_VERSION}" OUTPUT_VARIABLE
                                                                             lib_version1)
      list(TRANSFORM _RAPIDS_PKG_LIB_DEBUG_NAMES APPEND ".${_RAPIDS_VERSION}" OUTPUT_VARIABLE
                                                                              lib_version2)
      list(PREPEND _RAPIDS_PKG_LIB_DEBUG_NAMES ${lib_version1} ${lib_version2})
    endif()
  endif()

  if(DEFINED _RAPIDS_INITIAL_CODE_BLOCK)
    if(NOT DEFINED ${_RAPIDS_INITIAL_CODE_BLOCK})
      message(FATAL_ERROR "INITIAL_CODE_BLOCK variable `${_RAPIDS_INITIAL_CODE_BLOCK}` doesn't exist"
      )
    endif()
    set(_RAPIDS_FIND_INITIAL_CODE_BLOCK "${${_RAPIDS_INITIAL_CODE_BLOCK}}")
  endif()

  if(DEFINED _RAPIDS_FINAL_CODE_BLOCK)
    if(NOT DEFINED ${_RAPIDS_FINAL_CODE_BLOCK})
      message(FATAL_ERROR "FINAL_CODE_BLOCK variable `${_RAPIDS_FINAL_CODE_BLOCK}` doesn't exist")
    endif()
    set(_RAPIDS_FIND_FINAL_CODE_BLOCK "${${_RAPIDS_FINAL_CODE_BLOCK}}")
  endif()

  # Need to generate the module
  string(TIMESTAMP current_year "%Y" UTC)
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/find_module.cmake.in"
                 "${CMAKE_BINARY_DIR}/cmake/find_modules/Find${name}.cmake" @ONLY)

  # Need to add our generated modules to CMAKE_MODULE_PATH!
  if(NOT "${CMAKE_BINARY_DIR}/rapids-cmake/find_modules/" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/cmake/find_modules/")
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
  endif()

  # Record what export sets this module is part of
  include("${rapids-cmake-dir}/export/find_package_file.cmake")
  rapids_export_find_package_file(BUILD "${CMAKE_BINARY_DIR}/cmake/find_modules/Find${name}.cmake"
                                  EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
  rapids_export_find_package_file(INSTALL "${CMAKE_BINARY_DIR}/cmake/find_modules/Find${name}.cmake"
                                  EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
endfunction()

cmake_policy(POP)
