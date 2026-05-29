# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

# Function
function(rapids_export_parse_version rapids_version orig_prefix ver_value)
  include("${rapids-cmake-dir}/cmake/parse_version.cmake")

  set(orig_version ${rapids_version})
  rapids_cmake_parse_version(MAJOR "${rapids_version}" orig_major_version)
  rapids_cmake_parse_version(MINOR "${rapids_version}" orig_minor_version)
  rapids_cmake_parse_version(PATCH "${rapids_version}" orig_patch_version)

  set(version_compat SameMajorVersion)
  if(DEFINED orig_major_version)
    set(rapids_major_version "${orig_major_version}")
    if(rapids_major_version MATCHES "^0+$")
      set(rapids_major_version "0")
    endif()
    string(APPEND rapids_project_version "${rapids_major_version}")
  endif()

  if(DEFINED orig_minor_version)
    set(rapids_minor_version "${orig_minor_version}")
    if(rapids_minor_version MATCHES "^0+$")
      set(rapids_minor_version "0")
    endif()
    string(APPEND rapids_project_version ".${rapids_minor_version}")
    set(version_compat SameMinorVersion)
  endif()

  if(DEFINED orig_patch_version)
    set(rapids_patch_version "${orig_patch_version}")
    if(rapids_patch_version MATCHES "^0+$")
      set(rapids_patch_version "0")
    endif()
    string(APPEND rapids_project_version ".${rapids_patch_version}")
    set(version_compat SameMinorVersion)
  endif()

  set(${ver_value} ${rapids_project_version} PARENT_SCOPE)
  set(${ver_value}_compat ${version_compat} PARENT_SCOPE)

  set(${orig_prefix}_version ${rapids_version} PARENT_SCOPE)
  set(${orig_prefix}_major_version ${orig_major_version} PARENT_SCOPE)
  set(${orig_prefix}_minor_version ${orig_minor_version} PARENT_SCOPE)
  set(${orig_prefix}_patch_version ${orig_patch_version} PARENT_SCOPE)
endfunction()
