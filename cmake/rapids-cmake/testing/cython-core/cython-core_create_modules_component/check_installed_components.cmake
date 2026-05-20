# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

file(GLOB _install_component1_files ${CMAKE_CURRENT_BINARY_DIR}/install-component1/*.so)
file(GLOB _install_component2_files ${CMAKE_CURRENT_BINARY_DIR}/install-component2/*.so)

if(NOT _install_component1_files MATCHES "test1")
  message(FATAL_ERROR "test1 was not installed into install-component1")
endif()
if(_install_component1_files MATCHES "test2")
  message(FATAL_ERROR "test2 was installed into install-component1")
endif()
if(_install_component2_files MATCHES "test1")
  message(FATAL_ERROR "test1 was installed into install-component2")
endif()
if(NOT _install_component2_files MATCHES "test2")
  message(FATAL_ERROR "test2 was not installed into install-component2")
endif()
