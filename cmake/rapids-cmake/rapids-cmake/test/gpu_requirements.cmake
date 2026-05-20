# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_test_gpu_requirements
----------------------------

.. versionadded:: v23.04.00

States how many GPUs and what percent of each a test requires.

  .. code-block:: cmake

    rapids_test_gpu_requirements( test_name GPUS <N> [PERCENT <value>])

This function should only be used when :cmake:command:`rapids_test_add` is
insufficient due to the rapids-cmake test wrappers not working for your
project.

When combined with :cmake:command:`rapids_test_init` informs CTest what
resources should be allocated to a test so that when testing in parallel
oversubscription doesn't occur. Without this information user execution of
CTest with high parallel levels will cause multiple tests to run on the
same GPU and quickly exhaust all memory.


``GPUS``
  State how many GPUs this test requires. Allows CTest to not over-subscribe
  a machine's hardware.

  Any integer value >= 0 is supported

``PERCENT``
  State how much of each GPU this test requires. In general 100, 50, and 20
  are commonly used values. By default if no percent is provided, 100 is
  used.

  Any integer value >= 0 and <= 100 is supported

  Default value of 100

#]=======================================================================]
function(rapids_test_gpu_requirements test_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.gpu_requirements")

  set(options)
  set(one_value GPUS PERCENT)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(DEFINED _RAPIDS_TEST_PERCENT AND NOT DEFINED _RAPIDS_TEST_GPUS)
    message(FATAL_ERROR "rapids_test_gpu_requirements requires the GPUS option to be provided when PERCENT is"
    )
  endif()

  set(gpus 0)
  if(DEFINED _RAPIDS_TEST_GPUS)
    set(gpus ${_RAPIDS_TEST_GPUS})
  endif()

  set(percent 100)
  if(DEFINED _RAPIDS_TEST_PERCENT)
    set(percent ${_RAPIDS_TEST_PERCENT})
  endif()

  # verify that gpu and percent are within the allowed bounds
  if(NOT gpus GREATER_EQUAL 0)
    message(FATAL_ERROR "rapids_test_gpu_requirements requires a numeric GPUS value [0-N].")
  endif()
  if(NOT (percent GREATER_EQUAL 1 AND percent LESS_EQUAL 100))
    message(FATAL_ERROR "rapids_test_gpu_requirements requires a numeric PERCENT value [1-100].")
  endif()

  if(gpus AND percent)
    set_property(TEST ${test_name} PROPERTY RESOURCE_GROUPS "${gpus},gpus:${percent}")
    set_property(TEST ${test_name} PROPERTY FIXTURES_REQUIRED resource_spec)
  endif()

endfunction()
