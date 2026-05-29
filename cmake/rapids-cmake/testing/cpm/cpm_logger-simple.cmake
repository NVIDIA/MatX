# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)

rapids_cpm_init()
if(TARGET rapids_logger::rapids_logger)
  message(FATAL_ERROR "Expected rapids_logger::rapids_logger not to exist")
endif()

if(COMMAND create_logger_macros)
  message(FATAL_ERROR "Expected create_logger_macros function not to exist")
endif()

rapids_cpm_rapids_logger()

if(NOT TARGET rapids_logger::rapids_logger)
  message(FATAL_ERROR "Expected rapids_logger::rapids_logger to exist")
endif()

if(NOT COMMAND create_logger_macros)
  message(FATAL_ERROR "Expected create_logger_macros function to exist")
endif()
