# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)

rapids_cpm_init()
rapids_cpm_package_override("${CMAKE_CURRENT_BINARY_DIR}/../rapids-cmake/pinned_versions.json")

# Verify that each git_tag is now different.
rapids_cpm_package_details_internal(rmm rmm_version rmm_repository pin_rmm_tag rmm_src_dir
                                    rmm_shallow rmm_exclude)
rapids_cpm_package_details_internal(cuco cuco_version cuco_repository pin_cuco_tag cuco_src_dir
                                    cuco_shallow cuco_exclude)

if(NOT pin_rmm_tag STREQUAL "e3e32150d1bef13480ede4b2df1f0b38e43108a4")
  message(FATAL_ERROR "pinned rmm tag (${pin_rmm_tag}) doesn't match pre-computed SHA1 for the 22.02 tag"
  )
endif()
if(NOT pin_cuco_tag STREQUAL "f823d30d6b08a60383266db25821074dbdbe5822")
  message(FATAL_ERROR "pinned cuco tag (${pin_cuco_tag}) doesn't match hardcoded SHA1(f823d30...)")
endif()

# Verify that the json file has the inline patches.

function(validate_has_inline_patches package_name)
  include("${rapids-cmake-dir}/cpm/detail/get_override_json.cmake")
  get_override_json(${package_name} json_data)

  string(TOLOWER "${package_name}" normalized_pkg_name)
  get_property(override_json_path GLOBAL
               PROPERTY rapids_cpm_${normalized_pkg_name}_override_json_file)
  string(JSON json_data ERROR_VARIABLE no_override_patch GET "${json_data}" patches)

  # Gather number of patches
  string(JSON patch_count LENGTH "${json_data}")
  if(patch_count GREATER_EQUAL 1)
    math(EXPR patch_count "${patch_count} - 1")
    foreach(index RANGE ${patch_count})
      string(JSON patch_data GET "${json_data}" ${index})
      string(JSON value ERROR_VARIABLE have_error GET "${patch_data}" "inline_patch")
      if(have_error)
        message(FATAL_ERROR "${package_name} patch entry ${patch_count} wasn't an inline patch")
      endif()
    endforeach()
  else()
    message(FATAL_ERROR "Was expecting patches to exist, but none found for ${package_name}")
  endif()
endfunction()

validate_has_inline_patches(rmm)
validate_has_inline_patches(cuco)
