# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_package_details
--------------------------

. code-block:: cmake

  rapids_cpm_package_details(<package_name>
                             <version_variable>
                             <git_url_variable>
                             <git_tag_variable>
                             <shallow_variable>
                             <exclude_from_all_variable>
                             )

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`rapids_cmake_always_download` will contain the value of the `always_download` entry if it exists.
  :cmake:variable:`CPM_DOWNLOAD_ALL` will contain the value of the `always_download` entry if it exists.

#]=======================================================================]
# cmake-lint: disable=R0913
function(rapids_cpm_package_details package_name version_var url_var tag_var shallow_var
         exclude_from_all_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_details")

  include("${rapids-cmake-dir}/cmake/detail/policy.cmake")
  rapids_cmake_policy(DEPRECATED_IN 25.10
                      REMOVED_IN 26.02
                      MESSAGE [=[`rapids_cpm_package_details` is deprecated. Please use `rapids_cpm_package_info` instead.]=]
  )

  rapids_cpm_package_details_internal(${package_name} ${version_var} ${url_var} ${tag_var}
                                      src_subdir ${shallow_var} ${exclude_from_all_var})
  set(${version_var} ${${version_var}} PARENT_SCOPE)
  set(${url_var} ${${url_var}} PARENT_SCOPE)
  set(${tag_var} ${${tag_var}} PARENT_SCOPE)
  set(${shallow_var} ${${shallow_var}} PARENT_SCOPE)
  set(${exclude_from_all_var} ${${exclude_from_all_var}} PARENT_SCOPE)
  if(DEFINED rapids_cmake_always_download)
    set(rapids_cmake_always_download ${rapids_cmake_always_download} PARENT_SCOPE)
    set(CPM_DOWNLOAD_ALL ${CPM_DOWNLOAD_ALL} PARENT_SCOPE)
  endif()

endfunction()

# cmake-lint: disable=R0912,R0913,R0915
function(rapids_cpm_package_details_internal package_name version_var url_var tag_var subdir_var
         shallow_var exclude_from_all_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_package_details_internal")

  include("${rapids-cmake-dir}/cpm/detail/load_preset_versions.cmake")
  rapids_cpm_load_preset_versions()

  include("${rapids-cmake-dir}/cpm/detail/get_default_json.cmake")
  include("${rapids-cmake-dir}/cpm/detail/get_override_json.cmake")
  get_default_json(${package_name} json_data)
  get_override_json(${package_name} override_json_data)

  # Parse required fields
  function(rapids_cpm_json_get_value name)
    string(JSON value ERROR_VARIABLE have_error GET "${override_json_data}" ${name})
    if(have_error)
      string(JSON value ERROR_VARIABLE have_error GET "${json_data}" ${name})
    endif()

    if(NOT have_error)
      set(${name} ${value} PARENT_SCOPE)
    endif()
  endfunction()

  rapids_cpm_json_get_value(version)
  rapids_cpm_json_get_value(git_url)
  rapids_cpm_json_get_value(git_tag)
  rapids_cpm_json_get_value(url)
  rapids_cpm_json_get_value(url_hash)

  # Handle mode switching in overrides: if override provides git mode, don't inherit url mode from
  # default (and vice versa). This allows overrides to switch between git and url fetch modes.
  if(override_json_data)
    string(JSON value ERROR_VARIABLE no_git_url_in_override GET "${override_json_data}" git_url)
    string(JSON value ERROR_VARIABLE no_git_tag_in_override GET "${override_json_data}" git_tag)
    string(JSON value ERROR_VARIABLE no_url_in_override GET "${override_json_data}" url)
    string(JSON value ERROR_VARIABLE no_url_hash_in_override GET "${override_json_data}" url_hash)

    set(override_has_git_fields FALSE)
    set(override_has_url_fields FALSE)
    if(NOT no_git_url_in_override OR NOT no_git_tag_in_override)
      set(override_has_git_fields TRUE)
    endif()
    if(NOT no_url_in_override OR NOT no_url_hash_in_override)
      set(override_has_url_fields TRUE)
    endif()

    # If override provides git mode, clear any inherited url mode
    if(override_has_git_fields)
      unset(url)
      unset(url_hash)
    endif()
    # If override provides url mode, clear any inherited git mode
    if(override_has_url_fields)
      unset(git_url)
      unset(git_tag)
    endif()
  endif()

  # Only do validation if we have an entry
  if(json_data OR override_json_data)
    # Validate that we have the required fields Either (git_url + git_tag) OR (url + url_hash) must
    # be specified
    if(NOT version)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it is missing a `version` entry"
      )
    endif()

    # Check for incomplete git mode
    if(git_url AND NOT git_tag)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it has 'git_url' but is missing 'git_tag'"
      )
    endif()
    if(git_tag AND NOT git_url)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it has 'git_tag' but is missing 'git_url'"
      )
    endif()

    # Check for incomplete url mode
    if(url AND NOT url_hash)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it has 'url' but is missing 'url_hash'"
      )
    endif()
    if(url_hash AND NOT url)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it has 'url_hash' but is missing 'url'"
      )
    endif()

    set(has_git_mode FALSE)
    set(has_url_mode FALSE)
    if(git_url AND git_tag)
      set(has_git_mode TRUE)
    endif()
    if(url AND url_hash)
      set(has_url_mode TRUE)
    endif()

    if(has_git_mode AND has_url_mode)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it has both git_url/git_tag and url/url_hash. Only one mode is allowed."
      )
    elseif(NOT has_git_mode AND NOT has_url_mode)
      message(FATAL_ERROR "rapids_cmake can't parse '${package_name}' json entry, it must have either (git_url and git_tag) or (url and url_hash)"
      )
    endif()
  endif()

  if(override_json_data)
    string(JSON value ERROR_VARIABLE no_git_url_override GET "${override_json_data}" git_url)
    string(JSON value ERROR_VARIABLE no_git_tag_override GET "${override_json_data}" git_tag)
    string(JSON value ERROR_VARIABLE no_url_override GET "${override_json_data}" url)
    string(JSON value ERROR_VARIABLE no_url_hash_override GET "${override_json_data}" url_hash)
    string(JSON value ERROR_VARIABLE no_patches_override GET "${override_json_data}" patches)
    set(fetch_details_overridden TRUE)
    if(no_git_url_override
       AND no_git_tag_override
       AND no_url_override
       AND no_url_hash_override
       AND no_patches_override)
      set(fetch_details_overridden FALSE)
    endif()
  endif()

  # Parse optional fields, set the variable to the 'default' value first
  set(git_shallow ON)
  rapids_cpm_json_get_value(git_shallow)

  unset(source_subdir)
  rapids_cpm_json_get_value(source_subdir)

  set(exclude_from_all OFF)
  rapids_cpm_json_get_value(exclude_from_all)

  # Ensure that always_download is not set by default so that the if(DEFINED always_download) check
  # below works as expected in the default case.
  unset(always_download)
  unset(override_ignored)
  if(override_json_data AND json_data AND fetch_details_overridden)
    # `always_download` default value requires the package to exist in both the default and override
    # and that the git url / git tag have been modified. We also need to make sure that when using
    # an override that it isn't disabled due to `CPM_<pkg>_SOURCE`
    string(TOLOWER "${package_name}" normalized_pkg_name)
    get_property(override_ignored GLOBAL
                 PROPERTY rapids_cpm_${normalized_pkg_name}_override_ignored)
    if(NOT (override_ignored OR DEFINED CPM_${package_name}_SOURCE))
      set(always_download ON)
    endif()
  endif()
  rapids_cpm_json_get_value(always_download)

  # Evaluate any magic placeholders in the version or tag components including the
  # `rapids-cmake-version` and `rapids-cmake-checkout-tag` values
  include("${rapids-cmake-dir}/rapids-version.cmake")

  cmake_language(EVAL CODE "set(version ${version})")

  # Handle git mode vs url mode For git mode: set url_var to git_url, tag_var to git_tag For url
  # mode: set url_var to url, tag_var to empty, and set _rapids_url_hash in parent scope
  if(has_url_mode)
    cmake_language(EVAL CODE "set(url ${url})")
    set(${version_var} ${version} PARENT_SCOPE)
    set(${url_var} "${url}" PARENT_SCOPE)
    set(${tag_var} "" PARENT_SCOPE)
    set(_rapids_url_hash "${url_hash}" PARENT_SCOPE)
  else()
    cmake_language(EVAL CODE "set(git_tag ${git_tag})")
    cmake_language(EVAL CODE "set(git_url ${git_url})")
    set(${version_var} ${version} PARENT_SCOPE)
    set(${url_var} "${git_url}" PARENT_SCOPE)
    set(${tag_var} "${git_tag}" PARENT_SCOPE)
  endif()
  set(${shallow_var} ${git_shallow} PARENT_SCOPE)
  set(${exclude_from_all_var} ${exclude_from_all} PARENT_SCOPE)
  if(DEFINED source_subdir)
    set(${subdir_var} ${source_subdir} PARENT_SCOPE)
  endif()
  if(DEFINED always_download)
    set(rapids_cmake_always_download ${always_download} PARENT_SCOPE)
    set(CPM_DOWNLOAD_ALL ${always_download} PARENT_SCOPE)
  endif()

endfunction()
