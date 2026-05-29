# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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
    set(rapids_lang_lower cxx)
    # Even when the CUDA language is disabled we want to pass this since it is used by
    # find_package(CUDAToolkit) to find the location
    set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_CUDA_COMPILER)
  endif()
  if("CUDA" IN_LIST rapids_languages)
    set(rapids_lang CUDA)
    set(rapids_lang_lower cuda)
  endif()

  if(NOT rapids_lang)
    message(FATAL_ERROR "rapids_test_generate_resource_spec Requires the CUDA or C++ language to be enabled."
    )
  endif()

  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/default_names.cmake)
  set(eval_exe ${PROJECT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name})

  if(NOT TARGET generate_ctest_json)
    find_package(CUDAToolkit QUIET)

    add_executable(generate_ctest_json
                   ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_resource_spec.cpp)
    if(CUDAToolkit_FOUND)
      target_link_libraries(generate_ctest_json PRIVATE CUDA::cudart_static)
      target_compile_definitions(generate_ctest_json PRIVATE HAVE_CUDA)
    endif()
    set_property(SOURCE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_resource_spec.cpp
                 PROPERTY LANGUAGE ${rapids_lang})
    set_target_properties(generate_ctest_json
                          PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/rapids-cmake/"
                                     OUTPUT_NAME ${rapids_test_generate_exe_name})
    target_compile_features(generate_ctest_json PRIVATE ${rapids_lang_lower}_std_17)

    add_test(NAME generate_resource_spec COMMAND generate_ctest_json "${filepath}")
    set_tests_properties(generate_resource_spec
                         PROPERTIES FIXTURES_SETUP resource_spec GENERATED_RESOURCE_SPEC_FILE
                                    "${filepath}")
  endif()

endfunction()
