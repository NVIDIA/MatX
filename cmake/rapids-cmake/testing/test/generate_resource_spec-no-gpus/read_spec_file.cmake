# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/spec.json")
  message(FATAL_ERROR "rapids_test_generate_resource_spec failed to write out the requested spec file"
  )
endif()

file(READ "${CMAKE_CURRENT_BINARY_DIR}/spec.json" content)
if(NOT content MATCHES [=[.*{"id": "0", "slots": 0}.*]=])
  message(FATAL_ERROR "rapids_test_generate_resource_spec incorrectly detected a GPU")
endif()
