# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
file(STRINGS "${EXECUTABLE}" contents)

execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                WORKING_DIRECTORY ${WORKING_DIRECTORY}
                ERROR_QUIET
                OUTPUT_VARIABLE RAPIDS_WRITE_SHA1
                OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
)
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY ${WORKING_DIRECTORY}
                ERROR_QUIET
                OUTPUT_VARIABLE RAPIDS_WRITE_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --tag --dirty --always
                WORKING_DIRECTORY ${WORKING_DIRECTORY}
                ERROR_QUIET
                OUTPUT_VARIABLE RAPIDS_WRITE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE # need to strip off any newline
)

if(NOT contents MATCHES "sha1=${RAPIDS_WRITE_SHA1}")
  message(FATAL_ERROR "SHA1 not embedded")
endif()
if(NOT contents MATCHES "branch=${RAPIDS_WRITE_BRANCH}")
  message(FATAL_ERROR "branch name not embedded")
endif()
if(NOT contents MATCHES "version=${RAPIDS_WRITE_VERSION}")
  message(FATAL_ERROR "git version not embedded")
endif()
