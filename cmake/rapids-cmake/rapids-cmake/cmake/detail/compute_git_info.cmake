#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

configure_file("${TEMPLATE_FILE}" "${FILE_TO_WRITE}" @ONLY)
