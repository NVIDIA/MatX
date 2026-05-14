# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

cmake_minimum_required(VERSION 3.26...3.29)

#[=[
The goal of this script is to re-parse the `CTestTestfile`
and record each test and relevant properties for execution
once installed.

This is done as part of the install process so that CMake's
generator expressions have been fully evaluated, and therefore
we can support them in rapids-cmake-test

Since in CMake Script mode we have no support for `add_test`,
`set_tests_properties` and `subdirs` we write our own version
of those functions which will allow us re-parse the information
in the `CTestTestfile` when we include it.


]=]

# =============================================================================
# ============== Helper Function                          ====================
# =============================================================================

# Convert values from CMake properties so that any path build directory paths become re-rooted in
# the install tree
#
# cmake-lint: disable=W0106
function(convert_paths_to_install_dir prop_var)
  set(possible_build_path "${${prop_var}}")
  cmake_path(GET possible_build_path FILENAME name)

  get_property(install_loc GLOBAL PROPERTY ${name}_install)
  if(install_loc)
    get_property(build_locs GLOBAL PROPERTY ${name}_build)
    foreach(build_loc IN LISTS build_locs)
      if(build_loc STREQUAL possible_build_path)
        string(REPLACE "${build_loc}" "${install_loc}/${name}" install_value
                       "${possible_build_path}")
        break()
      endif()
    endforeach()
  else()
    string(REPLACE "${_RAPIDS_BUILD_DIR}" "\${CMAKE_INSTALL_PREFIX}" install_value
                   "${possible_build_path}")
  endif()
  set(${prop_var} "${install_value}" PARENT_SCOPE)
endfunction()

# Convert a list of `<NAME>=<VALUE>` entries
#
# cmake-lint: disable=W0105
function(find_and_convert_paths_from_var_list prop_var)
  set(test_env_vars_and_values "${${prop_var}}")
  foreach(env_tuple IN LISTS test_env_vars_and_values)
    string(REPLACE "\"" "" env_tuple "${env_tuple}")
    string(REPLACE "=" ";" env_tuple "${env_tuple}")

    list(GET env_tuple 0 env_var) # get the name
    list(LENGTH env_tuple length)
    if(length EQUAL 1)
      list(APPEND transformed_vars_and_values "${env_var}=")
    else()
      list(GET env_tuple 1 env_value) # get the value
      convert_paths_to_install_dir(env_value)
      list(APPEND transformed_vars_and_values "${env_var}=${env_value}")
    endif()
  endforeach()
  set(${prop_var} "${transformed_vars_and_values}" PARENT_SCOPE)
endfunction()

# =============================================================================
# ============== Function Overrides                       ====================
# =============================================================================

# Provide an `add_test` function signature since the built-in version doesn't exist in script mode
function(add_test name command)
  if(NOT name IN_LIST _RAPIDS_TESTS_TO_RUN)
    return()
  endif()

  string(APPEND test_file_content "add_test([=[${name}]=]")

  # Transform absolute path to relative install path
  cmake_path(GET command FILENAME cname)
  if(cname STREQUAL cmake)
    # rewrite the abs path to cmake to a relative version so that we don't care where cmake is
    set(command "cmake")
  else()
    get_property(install_loc GLOBAL PROPERTY ${name}_install)
    if(install_loc)
      get_property(build_locs GLOBAL PROPERTY ${name}_build)
      foreach(build_loc IN LISTS build_locs)
        if(build_loc STREQUAL command)
          string(REPLACE "${build_loc}" "${install_loc}/${name}" command "${command}")
          break()
        endif()
      endforeach()
    endif()
  endif()

  # convert paths from args to be re-rooted in the install tree
  set(args "${ARGN}")
  find_and_convert_paths_from_var_list(args)

  # convert args from a list to a single string that is space separated
  string(JOIN " " args ${args})

  string(APPEND test_file_content " \"${command}\" ${args})\n")
  set(test_file_content "${test_file_content}" PARENT_SCOPE)
endfunction()

# Provide a `set_tests_properties` function signature since the built-in version doesn't exist in
# script mode
function(set_tests_properties name)
  if(NOT name IN_LIST _RAPIDS_TESTS_TO_RUN)
    return()
  endif()

  set(options PROPERTIES)
  set(env_props ENVIRONMENT #
                ENVIRONMENT_MODIFICATION)
  set(one_value
      FIXTURES_CLEANUP #
      FIXTURES_REQUIRED #
      FIXTURES_SETUP #
      LABELS #
      MEASUREMENT #
      PASS_REGULAR_EXPRESSION #
      PROCESSOR_AFFINITY #
      PROCESSORS #
      REQUIRED_FILES #
      RESOURCE_GROUPS #
      RESOURCE_LOCK #
      RUN_SERIAL #
      SKIP_REGULAR_EXPRESSION #
      SKIP_RETURN_CODE #
      TIMEOUT #
      TIMEOUT_AFTER_MATCH #
      WILL_FAIL)
  set(multi_value_no_propagate
      _BACKTRACE_TRIPLES #
      ATTACHED_FILES #
      ATTACHED_FILES_ON_FAIL #
      WORKING_DIRECTORY)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}"
                        "${env_props};${multi_value_no_propagate}" ${ARGN})
  foreach(prop IN LISTS env_props)
    if(_RAPIDS_TEST_${prop})
      set(prop_value "${_RAPIDS_TEST_${prop}}")
      find_and_convert_paths_from_var_list(prop_value)
      string(APPEND test_prop_content
             "set_tests_properties([=[${name}]=] PROPERTIES ${prop} \"${prop_value}\")\n")
    endif()
  endforeach()

  foreach(prop IN LISTS one_value)
    if(_RAPIDS_TEST_${prop})
      set(prop_value "${_RAPIDS_TEST_${prop}}")
      convert_paths_to_install_dir(prop_value)
      string(APPEND test_prop_content
             "set_tests_properties([=[${name}]=] PROPERTIES ${prop} ${prop_value})\n")
    endif()
  endforeach()

  string(APPEND test_file_content "${test_prop_content}\n")
  set(test_file_content "${test_file_content}" PARENT_SCOPE)
endfunction()

# Provide a `subdirs` function signature since the built-in version doesn't exist in script mode
function(subdirs name)
  string(APPEND test_file_content "\n")
  if(EXISTS "${name}/CTestTestfile.cmake")
    include("${name}/CTestTestfile.cmake")
  endif()
  set(test_file_content "${test_file_content}" PARENT_SCOPE)
endfunction()

# =============================================================================
# ============== Parse Install Location Functions         ====================
# =============================================================================
function(extract_install_info)
  # remove the trailing `)` so that it doesn't get parsed as part of the file name
  string(REGEX REPLACE "\\)$" "" line "${ARGN}")

  # Leverate separate_arguments to parse a space-separated string into a list of items We use
  # `UNIX_COMMAND` as that means args are separated by unquoted whitespace ( single, and double
  # supported).
  separate_arguments(install_contents UNIX_COMMAND "${line}")

  set(options "file(INSTALL")
  set(one_value DESTINATION TYPE)
  set(multi_value FILES)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}"
                        ${install_contents})
  if(_RAPIDS_TEST_TYPE STREQUAL "EXECUTABLE" OR _RAPIDS_TEST_TYPE STREQUAL "SHARED_LIBRARY"
     OR _RAPIDS_TEST_TYPE STREQUAL "STATIC_LIBRARY" OR _RAPIDS_TEST_TYPE STREQUAL "OBJECT_LIBRARY")
    foreach(build_loc IN LISTS _RAPIDS_TEST_FILES)
      cmake_path(GET build_loc FILENAME name)
      set_property(GLOBAL PROPERTY ${name}_install ${_RAPIDS_TEST_DESTINATION})

      # For multi-config generators we will have multiple locations
      set_property(GLOBAL APPEND PROPERTY ${name}_build ${build_loc})
    endforeach()
  endif()
endfunction()

#
# Find all the cmake_install.cmake files in the install directory and parse them for install rules
function(determine_install_location_of_all_targets)
  file(GLOB_RECURSE install_rule_files "${_RAPIDS_PROJECT_DIR}/cmake_install.cmake")
  foreach(file IN LISTS install_rule_files)
    file(STRINGS "${file}" contents)

    set(parsing_file_command FALSE)
    set(file_command_contents)
    foreach(line IN LISTS contents)
      if(line MATCHES "INSTALL DESTINATION")
        # We found the first line of `file(INSTALL`
        set(parsing_file_command TRUE)
      endif()

      if(parsing_file_command)
        # Continue to add the lines of `file(INSTALL` till we hit the closing `)` That allows us to
        # support multiple line file commands
        string(APPEND command_contents "${line}")
        if(line MATCHES "\\)$")
          # We have all the lines for this file command, now parse it
          extract_install_info(${command_contents})

          # Reset to empty state for next `file(INSTALL)` command
          set(parsing_file_command FALSE)
          unset(command_contents)
        endif()
      endif()
    endforeach()
  endforeach()
endfunction()

# =============================================================================
# ============== Generate Install CTestTestfile            ====================
# =============================================================================
determine_install_location_of_all_targets()
# Setup the install location of `run_gpu_test`
set_property(GLOBAL PROPERTY run_gpu_test.cmake_install ".")
set_property(GLOBAL PROPERTY run_gpu_test.cmake_build
                             "${_RAPIDS_PROJECT_DIR}/rapids-cmake/./run_gpu_test.cmake")

include(${CMAKE_CURRENT_LIST_DIR}/default_names.cmake)
set(test_file_content
    "
set(CTEST_SCRIPT_DIRECTORY \".\")
set(CMAKE_INSTALL_PREFIX \"./${_RAPIDS_INSTALL_PREFIX}\")
add_test(generate_resource_spec ./${rapids_test_generate_exe_name} \"./${rapids_test_json_file_name}\")
execute_process(COMMAND ./${rapids_test_generate_exe_name} --cwd COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE _cwd)
set_tests_properties(generate_resource_spec PROPERTIES
  FIXTURES_SETUP resource_spec
  GENERATED_RESOURCE_SPEC_FILE \"\${_cwd}/${rapids_test_json_file_name}\"
)
\n\n
")

# will cause the above `add_test`, etc hooks to trigger filling up the contents of
# `test_file_content`
if(EXISTS "${_RAPIDS_BUILD_DIR}/CTestTestfile.cmake")
  # Support multi-generators by setting the CTest config mode to be equal to the install mode
  set(CTEST_CONFIGURATION_TYPE "${CMAKE_INSTALL_CONFIG_NAME}")

  # Too support tests added via gtest_discover_tests we need to tell GoogleTest we aren't in script
  # mode. This stops GoogleTestAddTests from thinking it is being used in a POST_BUILD manner and
  # should try and look for an undefined executable
  include(GoogleTest)
  set(CMAKE_SCRIPT_MODE_FILE OFF)

  include("${_RAPIDS_BUILD_DIR}/CTestTestfile.cmake")
endif()

file(WRITE "${test_launcher_file}" "${test_file_content}")
