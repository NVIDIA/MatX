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
rapids_cpm_nvcomp
-----------------
.. versionadded:: v22.06.00

Allow projects to find or build `nvComp` via `CPM` with built-in
tracking of these dependencies for correct export support.

Uses the version of nvComp :ref:`specified in the version file <cpm_versions>` for consistency
across all RAPIDS projects.

.. code-block:: cmake

  rapids_cpm_nvcomp( [USE_PROPRIETARY_BINARY <ON|OFF>]
                     [BUILD_EXPORT_SET <export-name>]
                     [INSTALL_EXPORT_SET <export-name>]
                     [<CPM_ARGS> ...])

``USE_PROPRIETARY_BINARY``
  By enabling this flag and using the software, you agree to fully comply with the terms and conditions of
  nvcomp's NVIDIA Software License Agreement. Found at https://developer.download.nvidia.com/compute/nvcomp/2.3/LICENSE.txt

  NVComp offers pre-built proprietary version of the library ( for x86_64 only ) that offer more features compared to the
  open source version. Since NVComp currently doesn't offer pre-built versions for all platforms, callers should verify
  the the request for a proprietary binary was fulfilled by checking the :cmake:variable:`nvcomp_proprietary_binary`
  variable after calling :cmake:command:`rapids_cpm_nvcomp`.

.. note::
  If an override entry exists for the nvcomp package it MUST have a proprietary_binary entry for this to
  flag to do anything. Any override without this entry is considered to invalidate the existin proprietary
  binary entry.

.. |PKG_NAME| replace:: nvcomp
.. include:: common_package_args.txt

Result Targets
^^^^^^^^^^^^^^
  nvcomp::nvcomp target will be created

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`nvcomp_SOURCE_DIR` is set to the path to the source directory of nvcomp.
  :cmake:variable:`nvcomp_BINARY_DIR` is set to the path to the build directory of nvcomp.
  :cmake:variable:`nvcomp_ADDED`      is set to a true value if nvcomp has not been added before.
  :cmake:variable:`nvcomp_VERSION`    is set to the version of nvcomp specified by the versions.json.
  :cmake:variable:`nvcomp_proprietary_binary` is set to ON if the proprietary binary is being used

#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_cpm_nvcomp)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.nvcomp")

  set(options)
  set(one_value USE_PROPRIETARY_BINARY BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()
  if(_RAPIDS_BUILD_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS BUILD_EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(nvcomp version repository tag shallow exclude)
  set(to_exclude OFF)
  if(NOT _RAPIDS_INSTALL_EXPORT_SET OR exclude)
    set(to_exclude ON)
  endif()

  # first see if we have a proprietary pre-built binary listed in versions.json and it if requested.
  set(nvcomp_proprietary_binary OFF) # will be set to true by rapids_cpm_get_proprietary_binary
  if(_RAPIDS_USE_PROPRIETARY_BINARY)
    include("${rapids-cmake-dir}/cpm/detail/get_proprietary_binary.cmake")
    rapids_cpm_get_proprietary_binary(nvcomp ${version})

    # Remove incorrect public dependency on the static cuda runtime We have to modify the
    # nvcomp-targets.cmake since these entries will cause a failure when rapids_cpm_find is called.
    if(nvcomp_proprietary_binary)
      set(target_file "${nvcomp_ROOT}/lib/cmake/nvcomp/nvcomp-targets.cmake")
      if(EXISTS "${target_file}")
        file(READ "${target_file}" file_contents)
        string(REPLACE "CUDA::cudart_static" "" file_contents "${file_contents}")
        file(WRITE "${target_file}" "${file_contents}")
      endif()
    endif()
  endif()

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(nvcomp ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS nvcomp::nvcomp
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow}
                  EXCLUDE_FROM_ALL ${to_exclude}
                  OPTIONS "BUILD_STATIC ON" "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
                          "BUILD_EXAMPLES OFF")

  # provice consistent targets between a found nvcomp and one building from source
  if(NOT TARGET nvcomp::nvcomp AND TARGET nvcomp)
    add_library(nvcomp::nvcomp ALIAS nvcomp)
  endif()

  # Propagate up variables that CPMFindPackage provide
  set(nvcomp_SOURCE_DIR "${nvcomp_SOURCE_DIR}" PARENT_SCOPE)
  set(nvcomp_BINARY_DIR "${nvcomp_BINARY_DIR}" PARENT_SCOPE)
  set(nvcomp_ADDED "${nvcomp_ADDED}" PARENT_SCOPE)
  set(nvcomp_VERSION ${version} PARENT_SCOPE)
  set(nvcomp_proprietary_binary ${nvcomp_proprietary_binary} PARENT_SCOPE)

  # Set up up install rules when using the proprietary_binary. When building from source, nvcomp
  # will set the correct install rules
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  if(_RAPIDS_INSTALL_EXPORT_SET AND nvcomp_proprietary_binary)
    include(GNUInstallDirs)
    install(DIRECTORY "${nvcomp_ROOT}/lib/" DESTINATION lib)
    install(DIRECTORY "${nvcomp_ROOT}/include/" DESTINATION include)
    # place the license information in the location that conda uses
    install(FILES "${nvcomp_ROOT}/NOTICE" DESTINATION info/ RENAME NVCOMP_NOTICE)
    install(FILES "${nvcomp_ROOT}/LICENSE" DESTINATION info/ RENAME NVCOMP_LICENSE)
  endif()

  if(_RAPIDS_BUILD_EXPORT_SET AND nvcomp_proprietary_binary)
    # point our consumers to where they can find the pre-built version
    rapids_export_find_package_root(BUILD nvcomp "${nvcomp_ROOT}" ${_RAPIDS_BUILD_EXPORT_SET})
  endif()

endfunction()
