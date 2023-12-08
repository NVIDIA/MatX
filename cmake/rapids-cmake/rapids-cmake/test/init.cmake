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
rapids_test_init
----------------

.. versionadded:: v23.04.00

Establish necessary components for CTest GPU resource allocation to allow
for parallel tests.

  .. code-block:: cmake

    rapids_test_init(  )

Generates a JSON resource specification file representing the machine's GPUs
using system introspection. This will allow CTest to schedule multiple
single-GPU or multi-GPU tests in parallel on multi-GPU machines.

For tests to execute correctly they will need to use the
:cmake:command:`rapids_test_add` to register GPU requirements:

.. code-block:: cmake

  enable_testing()
  include(rapids-test)
  rapids_test_init()

  add_executable( test_example test.cu )
  rapids_test_add(NAME single_gpu_alloc COMMAND test_example GPUS 1)
  rapids_test_add(NAME two_gpu_alloc COMMAND test_example GPUS 2)

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`CTEST_RESOURCE_SPEC_FILE` will be set to the generated
  JSON file if not already set

#]=======================================================================]
function(rapids_test_init)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.init")

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/default_names.cmake)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_resource_spec.cmake)
  set(rapids_test_spec_file "${PROJECT_BINARY_DIR}/${rapids_test_json_file_name}")

  rapids_test_generate_resource_spec(DESTINATION "${rapids_test_spec_file}")

  if(NOT CTEST_RESOURCE_SPEC_FILE)
    set(CTEST_RESOURCE_SPEC_FILE "${rapids_test_spec_file}" PARENT_SCOPE)
  endif()

endfunction()
