# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-project LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/detail/get_default_json.cmake)
include(${rapids-cmake-dir}/cpm/detail/get_override_json.cmake)
include(${rapids-cmake-dir}/cpm/detail/package_details.cmake)
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

if(NOT projects-to-verify)
  message(FATAL_ERROR "Failed to pass anything in projects-to-verify")
endif()

# Get the baseline before the override is added
foreach(proj IN LISTS projects-to-verify)
  rapids_cpm_package_details_internal(${proj} ${proj}_version ${proj}_repository ${proj}_tag
                                      ${proj}_src_dir ${proj}_shallow ${proj}_exclude)
endforeach()

# Verify that the pinned_versions is valid json by using it
rapids_cpm_package_override("${pinned_versions_file}")

# helper macro for gathering versions.json and pinned_versions.json content to be printed in error
# messages
macro(_get_json_data)
  get_default_json(${proj} default_versions)
  get_override_json(${proj} pinned_versions)
endmacro()

foreach(proj IN LISTS projects-to-verify)
  # Verify that each git_tag is now different.
  rapids_cpm_package_details_internal(${proj} ${proj}_version ${proj}_repository pin_${proj}_tag
                                      pin_${proj}_src_dir pin_${proj}_shallow ${proj}_exclude)
  # 160 bits for SHA-1, 256 bits for SHA-256
  string(LENGTH "${${proj}_tag}" tag_length)
  if(pin_${proj}_tag STREQUAL ${proj}_tag
     AND NOT (${proj}_tag MATCHES "^[0-9a-f]+$" AND (tag_length EQUAL 40 OR tag_length EQUAL 64)))
    _get_json_data()
    message(FATAL_ERROR "pinned ${proj} tag (${pin_${proj}_tag}) should differ compared to baseline ${${proj}_tag}\npinned_versions.json:\n${pinned_versions}\nversions.json:\n${default_versions}"
    )
  elseif(NOT pin_${proj}_tag STREQUAL ${proj}_tag
         AND (${proj}_tag MATCHES "^[0-9a-f]+$" AND (tag_length EQUAL 40 OR tag_length EQUAL 64)))
    _get_json_data()
    message(FATAL_ERROR "pinned ${proj} tag (${pin_${proj}_tag}) should be the same as baseline ${${proj}_tag}\npinned_versions.json:\n${pinned_versions}\nversions.json:\n${default_versions}"
    )
  endif()

  if(proj IN_LIST projects-with-subdirs AND NOT pin_${proj}_src_dir)
    message(FATAL_ERROR "pinned ${proj} should have a `source_subdir` entry")
  endif()

  # Everything should have shallow marked as false so that clones by SHA1 work
  if(pin_${proj}_shallow)
    _get_json_data()
    message(FATAL_ERROR "pin_${proj}_shallow is expected to be false, but got ${pin_${proj}_shallow}\npinned_versions.json:\n${pinned_versions}\nversions.json:\n${default_versions}"
    )
  endif()
endforeach()

foreach(proj IN LISTS projects-not-in-list)
  # Verify that we don't have entries for these
  rapids_cpm_package_details_internal(
    ${proj} pin_${proj}_version ${proj}_repository pin_${proj}_tag pin_${proj}_src_dir
    ${proj}_shallow ${proj}_exclude)
  if(pin_${proj}_version)
    message(FATAL_ERROR "${proj} shouldn't be in override, it was a system installed package")
  endif()
endforeach()
