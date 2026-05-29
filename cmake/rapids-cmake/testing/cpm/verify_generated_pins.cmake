# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(verify_generated_pins target_name)
  set(_rapids_options)
  set(_rapids_one_value PIN_FILE)
  set(_rapids_multi_value PROJECTS PROJECTS_NOT_EXIST SOURCE_SUBDIR_PROJECTS)
  cmake_parse_arguments(PARSE_ARGV 1 _RAPIDS "${_rapids_options}" "${_rapids_one_value}"
                        "${_rapids_multi_value}")

  if(NOT DEFINED _RAPIDS_PROJECTS)
    message(FATAL_ERROR "verify_generated_pins must be called with `PROJECTS` to verify")
  endif()
  if(NOT DEFINED _RAPIDS_PIN_FILE)
    set(_RAPIDS_PIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/rapids-cmake/pinned_versions.json")
  endif()

  message(STATUS "Verifying generated pins for projects: ${_RAPIDS_PROJECTS}")
  message(STATUS "Pin file: ${_RAPIDS_PIN_FILE}")
  message(STATUS "Source subdir projects: ${_RAPIDS_SOURCE_SUBDIR_PROJECTS}")
  message(STATUS "Projects not in list: ${_RAPIDS_PROJECTS_NOT_EXIST}")
  foreach(proj IN LISTS _RAPIDS_PROJECTS)
    if(NOT CPM_PACKAGE_${proj}_SOURCE_DIR)
      message(FATAL_ERROR "Attempting to verify a project ( ${proj} ) that was not cloned as part of this build"
      )
    endif()
  endforeach()

  add_custom_target(${target_name} ALL
                    COMMAND ${CMAKE_COMMAND}
                            "-S${CMAKE_CURRENT_FUNCTION_LIST_DIR}/verify_generated_pins/"
                            "-B${CMAKE_BINARY_DIR}/${target_name}_verify_build"
                            "-Drapids-cmake-dir=${rapids-cmake-dir}"
                            "-Dpinned_versions_file=${_RAPIDS_PIN_FILE}"
                            "-Dprojects-to-verify=${_RAPIDS_PROJECTS}"
                            "-Dprojects-with-subdirs=${_RAPIDS_SOURCE_SUBDIR_PROJECTS}"
                            "-Dprojects-not-in-list=${_RAPIDS_PROJECTS_NOT_EXIST}"
                    VERBATIM)

endfunction()
