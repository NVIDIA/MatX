# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
if(GIT_EXECUTABLE AND EXISTS "${GIT_EXECUTABLE}")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                  WORKING_DIRECTORY ${WORKING_DIRECTORY}
                  ERROR_QUIET
                  OUTPUT_VARIABLE _RAPIDS_WRITE_SHA1
                  OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
  )
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                  WORKING_DIRECTORY ${WORKING_DIRECTORY}
                  ERROR_QUIET
                  OUTPUT_VARIABLE _RAPIDS_WRITE_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
  )
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tag --dirty --always
                  WORKING_DIRECTORY ${WORKING_DIRECTORY}
                  ERROR_QUIET
                  OUTPUT_VARIABLE _RAPIDS_WRITE_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
  )
endif()

if(NOT _RAPIDS_WRITE_SHA1)
  set(_RAPIDS_WRITE_SHA1 "unknown")
endif()
if(NOT _RAPIDS_WRITE_BRANCH)
  set(_RAPIDS_WRITE_BRANCH "unknown")
endif()
if(NOT _RAPIDS_WRITE_VERSION)
  set(_RAPIDS_WRITE_VERSION "unknown")
endif()

set(_RAPIDS_GIT_IS_DIRTY 0)
if(_RAPIDS_WRITE_VERSION MATCHES dirty)
  set(_RAPIDS_GIT_IS_DIRTY 1)
endif()

string(TIMESTAMP current_year "%Y" UTC)
configure_file("${TEMPLATE_FILE}" "${FILE_TO_WRITE}" @ONLY)
