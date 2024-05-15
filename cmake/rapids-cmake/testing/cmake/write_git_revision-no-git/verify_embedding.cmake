#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.23.1)
file(STRINGS "${EXECUTABLE}" contents)

execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    ERROR_QUIET
    OUTPUT_VARIABLE RAPIDS_WRITE_SHA1
    OUTPUT_STRIP_TRAILING_WHITESPACE #need to strip off any newline
    )
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    ERROR_QUIET
    OUTPUT_VARIABLE RAPIDS_WRITE_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE #need to strip off any newline
    )
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tag --dirty --always
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    ERROR_QUIET
    OUTPUT_VARIABLE RAPIDS_WRITE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE #need to strip off any newline
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
