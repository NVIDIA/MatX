# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_test_record_install
--------------------------

.. versionadded:: v23.04.00

Record that the provided target should have install rules specified when
:cmake:command:`rapids_test_install_relocatable` is called with the given component.

  .. code-block:: cmake

    rapids_test_record_install(TARGET <name> COMPONENT <set>)

#]=======================================================================]
function(rapids_test_record_install)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.test.record_install")

  set(options)
  set(one_value TARGET COMPONENT)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  set(component ${_RAPIDS_TEST_COMPONENT})
  set_property(TARGET rapids_test_install_${component} APPEND PROPERTY TARGETS_TO_INSTALL
                                                                       "${_RAPIDS_TEST_TARGET}")
endfunction()
