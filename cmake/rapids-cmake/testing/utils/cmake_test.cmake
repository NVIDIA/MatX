# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

include(utils/cmake_detect_generators.cmake)

#[=======================================================================[.rst:
add_cmake_build_test
--------------------

.. versionadded:: v21.06.00

Generates a CMake test(s) for the given input CMake file or directory.
Determines the set of supported generators from the below list and
adds a test for each generator:
  - `Ninja`
  - `Ninja Multi-Config`
  - `Unix Makefiles`

.. code-block:: cmake

  add_cmake_build_test( (config|build|test|install)
                         <SourceOrDir>
                         [SERIAL]
                         [NO_DEV_ERRORS]
                         [NO_CPM_CACHE]
                         [NO_RAPIDS_CMAKE_HOOKS]
                         [SHOULD_FAIL <expected error message string>]
                      )

``config``
  Generate a CMake project and runs CMake config and generate
  step. Expects the test to raise an error to mark failures

``build``
  Generate and build a CMake project. The CMake configuration
  step is constructed as a setup fixture.
  Expects the test to either raise an error during config, or fail
  to build to mark failures

``test``
  Generate, build, and ctest a CMake project. The CMake config, and
  build steps are constructed as a setup fixture.
  Expects the underlying ctest execution to fail to mark failures

``install``
  - Not implemented

#]=======================================================================]
function(add_cmake_test mode source_or_dir)
  set(options SERIAL NO_DEV_ERRORS NO_CPM_CACHE NO_RAPIDS_CMAKE_HOOKS)
  set(one_value SHOULD_FAIL)
  set(multi_value)
  cmake_parse_arguments(RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  cmake_detect_generators(supported_generators nice_gen_names)

  string(TOLOWER ${mode} mode)
  cmake_path(GET source_or_dir STEM test_name_stem)

  # Determine if we are past a relative source file or a directory
  set(have_source_dir FALSE)
  if(IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${source_or_dir}")
    set(src_dir "${CMAKE_CURRENT_LIST_DIR}/${source_or_dir}/")
  elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${source_or_dir}")
    set(src_dir "${CMAKE_CURRENT_BINARY_DIR}/${test_name_stem}")
    set(test_cmake_file "${CMAKE_CURRENT_LIST_DIR}/${source_or_dir}")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/project_template.cmake.in"
                   "${src_dir}/CMakeLists.txt" @ONLY)
  else()
    message(FATAL_ERROR "Unable to find a file or directory named: ${source_or_dir}")
  endif()

  set(extra_configure_flags "-DCMAKE_MESSAGE_LOG_LEVEL=DEBUG")

  if(NOT RAPIDS_TEST_NO_RAPIDS_CMAKE_HOOKS)
    list(APPEND
         extra_configure_flags
         "-DCMAKE_PROJECT_INCLUDE_BEFORE=${PROJECT_SOURCE_DIR}/utils/emulate_fetching_rapids_cmake.cmake"
    )
  endif()
  if(NOT (RAPIDS_TEST_NO_DEV_ERRORS OR RAPIDS_TEST_DISABLE_DEV_ERRORS))
    list(APPEND extra_configure_flags "-Werror=dev")
  endif()
  if(DEFINED CPM_SOURCE_CACHE AND NOT RAPIDS_TEST_NO_CPM_CACHE)
    list(APPEND extra_configure_flags "-DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}")
  endif()
  if(DEFINED CPM_DOWNLOAD_LOCATION)
    list(APPEND extra_configure_flags "-DCPM_DOWNLOAD_LOCATION=${CPM_DOWNLOAD_LOCATION}")
  endif()
  if(PACKAGES_IN_CPM_CACHE)
    # Prevent ever finding preexisting built packages for those that we have in the cache.
    foreach(pkg ${PACKAGES_IN_CPM_CACHE})
      list(APPEND extra_configure_flags "-DCPM_DOWNLOAD_${pkg}=ON")
    endforeach()
  endif()

  foreach(generator gen_name IN ZIP_LISTS supported_generators nice_gen_names)

    set(test_name "${test_name_stem}-${gen_name}")
    set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${test_name}-build")

    if(mode STREQUAL "config")
      add_test(NAME ${test_name}
               COMMAND ${CMAKE_COMMAND} -S ${src_dir} -B ${build_dir} -G "${generator}"
                       ${extra_configure_flags} -Drapids-cmake-testing-dir=${PROJECT_SOURCE_DIR}
                       -Drapids-cmake-dir=${PROJECT_SOURCE_DIR}/../rapids-cmake)

    elseif(mode STREQUAL "build")
      add_test(NAME ${test_name}_configure
               COMMAND ${CMAKE_COMMAND} -S ${src_dir} -B ${build_dir} -G "${generator}"
                       ${extra_configure_flags} -Drapids-cmake-testing-dir=${PROJECT_SOURCE_DIR}
                       -Drapids-cmake-dir=${PROJECT_SOURCE_DIR}/../rapids-cmake)

      add_test(NAME ${test_name} COMMAND ${CMAKE_COMMAND} --build ${build_dir} -j3000)

      set_tests_properties(${test_name}_configure PROPERTIES FIXTURES_SETUP ${test_name})
      set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED ${test_name})
    elseif(mode STREQUAL "test")
      add_test(NAME ${test_name}_configure
               COMMAND ${CMAKE_COMMAND} -S ${src_dir} -B ${build_dir} -G "${generator}"
                       ${extra_configure_flags} -Drapids-cmake-testing-dir=${PROJECT_SOURCE_DIR}
                       -Drapids-cmake-dir=${PROJECT_SOURCE_DIR}/../rapids-cmake)

      add_test(NAME ${test_name}_build COMMAND ${CMAKE_COMMAND} --build ${build_dir} -j3)
      set_tests_properties(${test_name}_build PROPERTIES DEPENDS ${test_name}_configure)

      add_test(NAME ${test_name} COMMAND ${CMAKE_CTEST_COMMAND} -C Debug -j400 -VV
               WORKING_DIRECTORY ${build_dir})

      set_tests_properties(${test_name}_configure PROPERTIES FIXTURES_SETUP ${test_name})
      set_tests_properties(${test_name}_build PROPERTIES FIXTURES_SETUP ${test_name})
      set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED ${test_name})
    elseif(mode STREQUAL "install")
      message(FATAL_ERROR "install mode not yet implemented by add_cmake_build_test")
    else()
      message(FATAL_ERROR "${mode} mode not one of the valid modes (config|build|install) by add_cmake_build_test"
      )
    endif()

    if(RAPIDS_TEST_SERIAL)
      set_tests_properties(${test_name} PROPERTIES RUN_SERIAL ON)
      if(TEST ${test_name}_configure)
        message(STATUS "${test_name}_configure is serial")
        set_tests_properties(${test_name}_configure PROPERTIES RUN_SERIAL ON)
      endif()
      if(TEST ${test_name}_build)
        set_tests_properties(${test_name}_build PROPERTIES RUN_SERIAL ON)
      endif()
    endif()

    if(RAPIDS_TEST_SHOULD_FAIL)
      # Make sure we have a match
      set_tests_properties(${test_name} PROPERTIES WILL_FAIL ON)
      set_tests_properties(${test_name} PROPERTIES FAIL_REGULAR_EXPRESSION
                                                   "${RAPIDS_TEST_SHOULD_FAIL}")
    else()
      # Error out if we detect any CMake syntax warnings
      set_tests_properties(${test_name} PROPERTIES FAIL_REGULAR_EXPRESSION "Syntax Warning")
    endif()

    # Apply a label to the test based on the folder it is in and the generator used
    get_filename_component(label_name ${CMAKE_CURRENT_LIST_DIR} NAME_WE)
    string(TOLOWER "${label_name}" lower_case_label)
    set_tests_properties(${test_name} PROPERTIES LABELS "${lower_case_label};${gen_name}")
  endforeach()

endfunction()
