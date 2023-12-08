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
rapids_test_record_test_component
---------------------------------

.. versionadded:: v23.04.00

Record what component a test is part of

  .. code-block:: cmake

    rapids_test_record_test_component(NAME <name> COMPONENT <set>)

#]=======================================================================]
function(rapids_test_record_test_component)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.record_test_component")

  set(options)
  set(one_value NAME COMPONENT)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  set(component ${_RAPIDS_TEST_COMPONENT})
  if(NOT TARGET rapids_test_install_${component})
    add_library(rapids_test_install_${component} INTERFACE)
  endif()
  set_property(TARGET rapids_test_install_${component} APPEND PROPERTY "TESTS_TO_RUN"
                                                                       "${_RAPIDS_TEST_NAME}")
endfunction()
