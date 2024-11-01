#=============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
rapids_test_generate_resource_spec
----------------------------------

.. versionadded:: v23.04.00

Generates a JSON resource specification file representing the machine's GPUs
using system introspection.

  .. code-block:: cmake

    rapids_test_generate_resource_spec( DESTINATION filepath )

Generates a JSON resource specification file representing the machine's GPUs
using system introspection. This will allow CTest to schedule multiple
single-GPU tests in parallel on multi-GPU machines.

For the majority of projects :cmake:command:`rapids_test_init` should be used.
This command should be used directly projects that require multiple spec
files to be generated.

``DESTINATION``
  Location that the JSON output from the detection should be written to

.. note::
    Unlike rapids_test_init this doesn't set CTEST_RESOURCE_SPEC_FILE

#]=======================================================================]
function(rapids_test_generate_resource_spec DESTINATION filepath)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.generate_resource_spec")

  unset(rapids_lang)
  get_property(rapids_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CXX" IN_LIST rapids_languages)
    set(rapids_lang CXX)
    # Even when the CUDA language is disabled we want to pass this since it is used by
    # find_package(CUDAToolkit) to find the location
    set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_CUDA_COMPILER)
  endif()
  if("CUDA" IN_LIST rapids_languages)
    set(rapids_lang CUDA)
  endif()

  if(NOT rapids_lang)
    message(FATAL_ERROR "rapids_test_generate_resource_spec Requires the CUDA or C++ language to be enabled."
    )
  endif()

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/default_names.cmake)
  set(eval_exe ${PROJECT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name})

  if(NOT EXISTS "${eval_exe}")
    find_package(CUDAToolkit QUIET)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/rapids-cmake/")

    try_compile(result "${PROJECT_BINARY_DIR}/rapids-cmake/generate_ctest_json-build"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_resource_spec"
                generate_resource_spec
                CMAKE_FLAGS "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}" "-Doutput_file=${eval_exe}"
                            "-Dlang=${rapids_lang}" "-Dcuda_toolkit=${CUDAToolkit_FOUND}"
                OUTPUT_VARIABLE compile_output)

    if(NOT result)
      string(REPLACE "\n" "\n  " compile_output "${compile_output}")
      message(FATAL_ERROR "rapids_test_generate_resource_spec failed to build detection executable.\nfailure details are:\n  ${compile_output}"
      )
    endif()
  endif()
  execute_process(COMMAND ${eval_exe} OUTPUT_FILE "${filepath}" COMMAND_ERROR_IS_FATAL ANY)

endfunction()
