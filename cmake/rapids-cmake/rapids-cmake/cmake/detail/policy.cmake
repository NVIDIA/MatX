#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cmake_policy
-------------------

.. versionadded:: v23.02.00

Prints rapids-cmake deprecated warnings

.. code-block:: cmake

  rapids_cmake_policy( DEPRECATED_IN <version> REMOVED_IN <version> MESSAGE <content>)

#]=======================================================================]
function(rapids_cmake_policy)
  set(options "")
  set(one_value DEPRECATED_IN REMOVED_IN MESSAGE)
  set(multi_value "")
  cmake_parse_arguments(_RAPIDS_POLICY "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED rapids-cmake-version)
    include("${rapids-cmake-dir}/rapids-version.cmake")
  endif()
  set(_RAPIDS_POLICY_CALLERS_VERSION ${rapids-cmake-version})
  set(policy_context_text
      "rapids-cmake policy [deprecated=${_RAPIDS_POLICY_DEPRECATED_IN} removed=${_RAPIDS_POLICY_REMOVED_IN}]:"
  )
  set(policy_mode DEPRECATION)
  message(STATUS "_RAPIDS_POLICY_CALLERS_VERSION: ${_RAPIDS_POLICY_CALLERS_VERSION}")
  message(STATUS "_RAPIDS_POLICY_REMOVED_IN: ${_RAPIDS_POLICY_REMOVED_IN}")
  if(_RAPIDS_POLICY_CALLERS_VERSION VERSION_GREATER_EQUAL ${_RAPIDS_POLICY_REMOVED_IN})
    set(policy_mode FATAL_ERROR)
  endif()

  set(policy_upgrade_text "")
  if(_RAPIDS_POLICY_CALLERS_VERSION VERSION_LESS ${_RAPIDS_POLICY_DEPRECATED_IN})
    set(policy_upgrade_text
        "You are currently requesting rapids-cmake ${_RAPIDS_POLICY_CALLERS_VERSION} please upgrade to ${_RAPIDS_POLICY_DEPRECATED_IN}."
    )
  endif()
  message(${policy_mode} "${policy_context_text} ${_RAPIDS_POLICY_MESSAGE} ${policy_upgrade_text}")

endfunction()
