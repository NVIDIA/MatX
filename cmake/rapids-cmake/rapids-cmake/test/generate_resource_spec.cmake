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

  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "rapids_test_generate_resource_spec Requires the CUDA or C++ language to be enabled."
    )
  endif()

  set(gpu_json_contents
      [=[
{
"version": {"major": 1, "minor": 0},
"local": [{
  "gpus": [{"id":"0", "slots": 0}]
}]
}
]=])

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/default_names.cmake)
  set(eval_file ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_resource_spec.cpp)
  set(eval_exe ${PROJECT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name})
  set(error_file ${PROJECT_BINARY_DIR}/rapids-cmake/detect_gpus.stderr.log)

  if(NOT EXISTS "${eval_exe}")
    find_package(CUDAToolkit QUIET)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/rapids-cmake/")

    if(CUDAToolkit_FOUND)
      set(compile_options "-I${CUDAToolkit_INCLUDE_DIRS}" "-DHAVE_CUDA")
    endif()
    set(link_options ${CUDA_cudart_LIBRARY})
    set(compiler "${CMAKE_CXX_COMPILER}")
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      set(compiler "${CMAKE_CUDA_COMPILER}")
    endif()

    execute_process(COMMAND "${compiler}" "${eval_file}" ${compile_options} ${link_options} -o
                            "${eval_exe}" OUTPUT_VARIABLE compile_output
                    ERROR_VARIABLE compile_output)
  endif()

  if(NOT EXISTS "${eval_exe}")
    message(STATUS "rapids_test_generate_resource_spec failed to build detection executable, presuming no GPUs."
    )
    message(STATUS "rapids_test_generate_resource_spec compile[${compiler} ${compile_options} ${link_options}] failure details are ${compile_output}"
    )
    file(WRITE "${filepath}" "${gpu_json_contents}")
  else()
    execute_process(COMMAND ${eval_exe} OUTPUT_FILE "${filepath}")
  endif()

endfunction()
