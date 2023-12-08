#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
rapids_test_install_relocatable
-------------------------------

.. versionadded:: v23.04.00

Install the needed `ctest` infrastructure to allow installed tests to be run
by `ctest` in parallel with GPU awareness.

  .. code-block:: cmake

    rapids_test_install_relocatable(INSTALL_COMPONENT_SET <component>
                                    DESTINATION <relative_path>
                                    [INCLUDE_IN_ALL])

Will install all tests created by :cmake:command:`rapids_test_add` that are
part of the provided ``INSTALL_COMPONENT_SET``.

The :cmake:command:`rapids_test_install_relocatable` will transform all
test arguments or properties on install tests that reference the build directory to reference
the install directory.

``INSTALL_COMPONENT_SET``
  Record which test component infrastructure to be installed

``DESTINATION``
  Relative path from the `CMAKE_INSTALL_PREFIX` to install the infrastructure.
  This needs to be the same directory as the test executables

``INCLUDE_IN_ALL``
  State that these install rules should be part of the default install set.
  By default tests are not part of the default install set.

.. note::
  rapids_test_install_relocatable behavior is undefined when used with
  multi-config generators such as "Visual Studio" and "Ninja Multi-Config"

#]=======================================================================]
function(rapids_test_install_relocatable)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.install_relocatable")

  set(options INCLUDE_IN_ALL)
  set(one_value INSTALL_COMPONENT_SET DESTINATION)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  set(to_exclude EXCLUDE_FROM_ALL)
  if(_RAPIDS_TEST_INCLUDE_IN_ALL)
    set(to_exclude)
  endif()

  set(component ${_RAPIDS_TEST_INSTALL_COMPONENT_SET})
  if(NOT TARGET rapids_test_install_${component})
    message(FATAL_ERROR "No install component set [${component}] can be found")
  endif()

  get_target_property(targets_to_install rapids_test_install_${component} TARGETS_TO_INSTALL)
  get_target_property(tests_to_run rapids_test_install_${component} TESTS_TO_RUN)

  string(REGEX REPLACE "/" ";" from_install_prefix "${_RAPIDS_TEST_DESTINATION}")
  list(TRANSFORM from_install_prefix REPLACE ".+" "../")
  list(JOIN from_install_prefix "" from_install_prefix)

  # cmake-lint: disable=W0106
  install(CODE "
    # set variables needed by `generate_installed_CTestTestfile.cmake`
    set(_RAPIDS_TEST_DESTINATION \"${_RAPIDS_TEST_DESTINATION}\")
    set(_RAPIDS_INSTALL_PREFIX \"${from_install_prefix}\")
    set(_RAPIDS_BUILD_DIR \"${CMAKE_CURRENT_BINARY_DIR}\")
    set(_RAPIDS_PROJECT_DIR \"${CMAKE_BINARY_DIR}\")
    set(_RAPIDS_INSTALL_COMPONENT_SET \"${_RAPIDS_TEST_INSTALL_COMPONENT_SET}\")
    set(_RAPIDS_TARGETS_INSTALLED \"${targets_to_install}\")
    set(_RAPIDS_TESTS_TO_RUN \"${tests_to_run}\")
    set(test_launcher_file \"${CMAKE_CURRENT_BINARY_DIR}/rapids-cmake/testing/CTestTestfile.cmake.to_install\")

    # parse the CTestTestfile and store installable version in `test_launcher_file`
    message(STATUS \"Generating install version of CTestTestfile.cmake\")
    cmake_policy(SET CMP0011 NEW) # PUSH/POP for `include`
    include(\"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_installed_CTestTestfile.cmake\")

    # install `test_launcher_file`
    file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_RAPIDS_TEST_DESTINATION}\" TYPE FILE RENAME \"CTestTestfile.cmake\" FILES \"\${test_launcher_file}\")
    "
          COMPONENT ${_RAPIDS_TEST_INSTALL_COMPONENT_SET}
          ${to_exclude})

  # We need to install the rapids-test gpu detector, and the json script we also need to write out /
  # install the new CTestTestfile.cmake
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/default_names.cmake)
  if(EXISTS "${PROJECT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name}")
    install(PROGRAMS "${PROJECT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name}"
            COMPONENT ${_RAPIDS_TEST_INSTALL_COMPONENT_SET} DESTINATION ${_RAPIDS_TEST_DESTINATION}
            ${to_exclude})
  endif()
  if(EXISTS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/run_gpu_test.cmake")
    install(FILES "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/run_gpu_test.cmake"
            COMPONENT ${_RAPIDS_TEST_INSTALL_COMPONENT_SET} DESTINATION ${_RAPIDS_TEST_DESTINATION}
            ${to_exclude})
  endif()
  if(targets_to_install)
    install(TARGETS ${targets_to_install} COMPONENT ${_RAPIDS_TEST_INSTALL_COMPONENT_SET}
            DESTINATION ${_RAPIDS_TEST_DESTINATION} ${to_exclude})
  endif()
endfunction()
