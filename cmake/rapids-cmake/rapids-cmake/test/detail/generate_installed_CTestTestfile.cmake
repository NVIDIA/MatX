#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
    get_property(build_loc GLOBAL PROPERTY ${name}_build)
    string(REPLACE "${build_loc}" "${install_loc}/${name}" install_value "${possible_build_path}")
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
      get_property(build_loc GLOBAL PROPERTY ${name}_build)
      string(REPLACE "${build_loc}" "${install_loc}/${name}" command "${command}")
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
function(extract_install_info line)
  # We have a problem where we want to split on spaces but only when it is between two UPPER CASE
  # letters and not in quotes :( We can't use string(REPLACE " ") since that will split paths with
  # spaces

  # what we can do is split on quotes and make this in a list. At that point we can split all other
  # entries again
  string(REPLACE "\"" ";" line "${line}")
  # item 1 is the install location item 2 is the filter if valid item 3+ are the lists of files
  # being installed
  list(GET line 2 type)
  if(type MATCHES " TYPE EXECUTABLE " OR type MATCHES " TYPE SHARED_LIBRARY "
     OR type MATCHES " TYPE STATIC_LIBRARY " OR type MATCHES " TYPE OBJECT_LIBRARY ")
    list(GET line 1 install_loc)
    list(GET line 3 build_loc)
    cmake_path(GET build_loc FILENAME name)
    set_property(GLOBAL PROPERTY ${name}_install ${install_loc})
    set_property(GLOBAL PROPERTY ${name}_build ${build_loc})
  endif()
endfunction()

#
# Find all the cmake_install.cmake files in the install directory and parse them for install rules
function(determine_install_location_of_all_targets)
  file(GLOB_RECURSE install_rule_files "${_RAPIDS_PROJECT_DIR}/cmake_install.cmake")
  foreach(file IN LISTS install_rule_files)
    file(STRINGS "${file}" contents REGEX "INSTALL DESTINATION")
    foreach(line IN LISTS contents)
      extract_install_info("${line}")
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
set(CTEST_RESOURCE_SPEC_FILE \"./${rapids_test_json_file_name}\")
execute_process(COMMAND ./${rapids_test_generate_exe_name} OUTPUT_FILE \"\${CTEST_RESOURCE_SPEC_FILE}\" COMMAND_ERROR_IS_FATAL ANY)
\n\n
")

# will cause the above `add_test`, etc hooks to trigger filling up the contents of
# `test_file_content`
if(EXISTS "${_RAPIDS_BUILD_DIR}/CTestTestfile.cmake")
  # Support multi-generators by setting the CTest config mode to be equal to the install mode
  set(CTEST_CONFIGURATION_TYPE "${CMAKE_INSTALL_CONFIG_NAME}")
  include("${_RAPIDS_BUILD_DIR}/CTestTestfile.cmake")
endif()

file(WRITE "${test_launcher_file}" "${test_file_content}")
