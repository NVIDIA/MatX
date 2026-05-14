# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(NOT DEFINED DEPFILE)
  message(FATAL_ERROR "Must pass DEPFILE")
endif()

if(NOT EXISTS "${DEPFILE}")
  message(FATAL_ERROR "rapids_cython_create_modules didn't create the dependency file. "
                      "Expected dependency file: ${DEPFILE}")
endif()
