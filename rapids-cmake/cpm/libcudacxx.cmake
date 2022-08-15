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
rapids_cpm_libcudacxx
---------------------

.. versionadded:: v21.12.00

Allow projects to find or build `libcudacxx` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of libcudacxx :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_libcudacxx( [BUILD_EXPORT_SET <export-name>]
                         [INSTALL_EXPORT_SET <export-name>]
                         [<CPM_ARGS> ...])

.. |PKG_NAME| replace:: libcudacxx
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  libcudacxx::libcudacxx target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`libcudacxx_SOURCE_DIR` is set to the path to the source directory of libcudacxx.
  :cmake:variable:`libcudacxx_BINARY_DIR` is set to the path to the build directory of  libcudacxx.
  :cmake:variable:`libcudacxx_ADDED`      is set to a true value if libcudacxx has not been added before.
  :cmake:variable:`libcudacxx_VERSION`    is set to the version of libcudacxx specified by the versions.json.

#]=======================================================================]
function(rapids_cpm_libcudacxx)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.libcudacxx")

  set(options CPM_ARGS)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(libcudacxx version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(libcudacxx ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS libcudacxx::libcudacxx
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow}
                  EXCLUDE_FROM_ALL ${exclude}
                  DOWNLOAD_ONLY TRUE)

  if(_RAPIDS_BUILD_EXPORT_SET)
    include("${rapids-cmake-dir}/export/package.cmake")
    rapids_export_package(BUILD libcudacxx ${_RAPIDS_BUILD_EXPORT_SET}
                          GLOBAL_TARGETS libcudacxx::libcudacxx)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD libcudacxx [=[${CMAKE_CURRENT_LIST_DIR}]=]
                                    ${_RAPIDS_BUILD_EXPORT_SET})
  endif()

  if(_RAPIDS_INSTALL_EXPORT_SET)
    include("${rapids-cmake-dir}/export/package.cmake")
    rapids_export_package(INSTALL libcudacxx ${_RAPIDS_INSTALL_EXPORT_SET} VERSION ${version}
                          GLOBAL_TARGETS libcudacxx::libcudacxx)
  endif()

  # establish the correct libcudacxx namespace aliases
  if(NOT TARGET rapids_libcudacxx AND NOT TARGET libcudacxx::libcudacxx AND libcudacxx_SOURCE_DIR)
    add_library(rapids_libcudacxx INTERFACE)
    set_target_properties(rapids_libcudacxx PROPERTIES EXPORT_NAME libcudacxx)

    add_library(libcudacxx::libcudacxx ALIAS rapids_libcudacxx)

    target_include_directories(rapids_libcudacxx
                               INTERFACE $<BUILD_INTERFACE:${libcudacxx_SOURCE_DIR}/include>
                                         $<INSTALL_INTERFACE:include/rapids/libcudacxx>)

    install(TARGETS rapids_libcudacxx DESTINATION ${lib_dir} EXPORT libcudacxx-targets)

    set(code_string
        [=[
# nvcc automatically adds the CUDA Toolkit system include paths before any
# system include paths that CMake adds. CMake implicitly treats all includes
# on import targets as 'SYSTEM' includes.
#
# To get this cudacxx to be picked up by consumers instead of the version shipped
# with the CUDA Toolkit we need to make sure it is a non-SYSTEM include on the CMake side.
#
if(NOT TARGET libcudacxx_includes)
  add_library(libcudacxx_includes INTERFACE)
  get_target_property(all_includes libcudacxx::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(libcudacxx::libcudacxx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
  set_target_properties(libcudacxx_includes PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${all_includes}")
  target_link_libraries(libcudacxx::libcudacxx INTERFACE libcudacxx_includes)
endif()
    ]=])

    if(_RAPIDS_BUILD_EXPORT_SET)
      include("${rapids-cmake-dir}/export/export.cmake")
      rapids_export(BUILD libcudacxx
                    EXPORT_SET libcudacxx-targets
                    GLOBAL_TARGETS libcudacxx
                    VERSION ${version}
                    NAMESPACE libcudacxx::
                    FINAL_CODE_BLOCK code_string)
    endif()

    if(_RAPIDS_INSTALL_EXPORT_SET)
      include(GNUInstallDirs) # For CMAKE_INSTALL_INCLUDEDIR
      install(DIRECTORY "${libcudacxx_SOURCE_DIR}/include/"
              DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/libcudacxx")
      install(DIRECTORY "${libcudacxx_SOURCE_DIR}/libcxx/include/"
              DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rapids/libcxx/include")

      include("${rapids-cmake-dir}/export/export.cmake")
      rapids_export(INSTALL libcudacxx
                    EXPORT_SET libcudacxx-targets
                    GLOBAL_TARGETS libcudacxx
                    VERSION ${version}
                    NAMESPACE libcudacxx::
                    FINAL_CODE_BLOCK code_string)

    endif()
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(libcudacxx_SOURCE_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
  set(libcudacxx_BINARY_DIR "${libcudacxx_BINARY_DIR}" PARENT_SCOPE)
  set(libcudacxx_ADDED "${libcudacxx_ADDED}" PARENT_SCOPE)
  set(libcudacxx_VERSION ${version} PARENT_SCOPE)

endfunction()
