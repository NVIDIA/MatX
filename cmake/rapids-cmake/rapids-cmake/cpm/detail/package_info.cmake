# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_package_info
-----------------------------

  rapids_cpm_package_info(<package_name> args....
                           VERSION_VAR <output_var_name>
                           FIND_VAR <output_var_name>
                           CPM_VAR <output_var_name>
                           TO_INSTALL_VAR <output_var_name>
                           )

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`rapids_cmake_always_download` will contain the value of the `always_download` entry if it exists.
  :cmake:variable:`CPM_DOWNLOAD_ALL` will contain the value of the `always_download` entry if it exists.
  :cmake:variable:`_RAPIDS_BUILD_EXPORT_SET` will contain the value of the BUILD_EXPORT entry if it exists.
  :cmake:variable:`_RAPIDS_INSTALL_EXPORT_SET` will contain the value of the INSTALL_EXPORT entry if it exists.


#]=======================================================================]
# cmake-lint: disable=R0915
function(rapids_cpm_package_info package_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_info")

  set(options FOR_FETCH_CONTENT)
  set(one_value VERSION_VAR FIND_VAR CPM_VAR TO_INSTALL_VAR BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Get all the package details
  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details_internal(${package_name} _rapids_version _rapids_url _rapids_tag
                                      _rapids_src_subdir _rapids_shallow _rapids_exclude_from_all)

  # Compute all the patch related details
  if(_RAPIDS_CPM_VAR OR _RAPIDS_FIND_VAR)
    include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
    rapids_cpm_generate_patch_command(${package_name} ${_rapids_version} patch_command
                                      build_patch_only)

    # The find info is the input unparsed args plus `build_patch_only`
    set(_rapids_find_content ${_RAPIDS_UNPARSED_ARGUMENTS} ${build_patch_only})

    # CMake has an issue where setting `GIT_SHALLOW` in both the declare and
    # FetchContent_MakeAvailable causes issues since the value becomes `OFF;OFF` which is a True
    # value.
    #
    # The cpm content is all the boiler plate info CPM needs such as GIT_REPO GIT_TAG GIT_SHALLOW
    # PATCH details SOURCE_SUBDIR -> todo EXCLUDE_FROM_ALL
    if(_rapids_tag)
      set(_rapids_cpm_content "GIT_REPOSITORY" "${_rapids_url}" "GIT_TAG" "${_rapids_tag}")
      if(NOT _RAPIDS_FOR_FETCH_CONTENT)
        list(APPEND _rapids_cpm_content "GIT_SHALLOW" "${_rapids_shallow}")
      endif()
    else()
      # When _rapids_tag is empty, we're using URL-based fetching (tarball) instead of git
      set(_rapids_cpm_content "URL" "${_rapids_url}")
      if(_rapids_url_hash)
        list(APPEND _rapids_cpm_content "URL_HASH" "${_rapids_url_hash}")
      endif()
    endif()

    if(NOT _RAPIDS_FOR_FETCH_CONTENT)
      list(APPEND _rapids_cpm_content "EXCLUDE_FROM_ALL" "${_rapids_exclude_from_all}")
    endif()
    if(patch_command)
      list(APPEND _rapids_cpm_content "${patch_command}")
    endif()
    if(_rapids_src_subdir)
      list(APPEND _rapids_cpm_content "SOURCE_SUBDIR" "${_rapids_src_subdir}")
    endif()
  endif()

  if(DEFINED _RAPIDS_BUILD_EXPORT_SET)
    list(APPEND _rapids_find_content "BUILD_EXPORT_SET" ${_RAPIDS_BUILD_EXPORT_SET})
  endif()
  if(DEFINED _RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _rapids_find_content "INSTALL_EXPORT_SET" ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()

  # The to_install logic is the combo of `_RAPIDS_INSTALL_EXPORT_SET` and `_rapids_exclude_from_all`
  set(_rapids_to_install OFF)
  if(_RAPIDS_INSTALL_EXPORT_SET AND NOT _rapids_exclude_from_all)
    set(_rapids_to_install ON)
  endif()

  set(${_RAPIDS_VERSION_VAR} ${_rapids_version} PARENT_SCOPE)
  set(${_RAPIDS_FIND_VAR} ${_rapids_find_content} PARENT_SCOPE)
  set(${_RAPIDS_CPM_VAR} ${_rapids_cpm_content} PARENT_SCOPE)
  set(${_RAPIDS_TO_INSTALL_VAR} ${_rapids_to_install} PARENT_SCOPE)
  set(_RAPIDS_BUILD_EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET} PARENT_SCOPE)
  set(_RAPIDS_INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET} PARENT_SCOPE)
  if(DEFINED rapids_cmake_always_download)
    set(rapids_cmake_always_download ${rapids_cmake_always_download} PARENT_SCOPE)
    set(CPM_DOWNLOAD_ALL ${CPM_DOWNLOAD_ALL} PARENT_SCOPE)
  endif()

endfunction()
