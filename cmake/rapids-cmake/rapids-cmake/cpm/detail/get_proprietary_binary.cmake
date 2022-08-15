#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
get_proprietary_binary
-------------------

.. versionadded:: v22.06.00

Download the associated proprietary binary for the given project based on
the current CPU target architecture ( x86_64, aarch64, etc )

 .. note::
  if override => the proprietary entry only in the override will be evaluated
  if no override => the proprietary entry only in the default will be evaluated


#]=======================================================================]
function(rapids_cpm_get_proprietary_binary package_name version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_get_proprietary_binary")

  include("${rapids-cmake-dir}/cpm/detail/get_default_json.cmake")
  include("${rapids-cmake-dir}/cpm/detail/get_override_json.cmake")
  get_default_json(${package_name} json_data)
  get_override_json(${package_name} override_json_data)

  # need to search the `proprietary_binary` dictionary for a key with the same name as
  # lower_case(`CMAKE_SYSTEM_PROCESSOR-CMAKE_SYSTEM_NAME`).
  set(key "${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}")
  string(TOLOWER ${key} key)
  if(override_json_data)
    string(JSON proprietary_binary ERROR_VARIABLE have_error GET "${override_json_data}"
           "proprietary_binary" "${key}")
  else()
    string(JSON proprietary_binary ERROR_VARIABLE have_error GET "${json_data}"
           "proprietary_binary" "${key}")
  endif()

  if(have_error)
    message(VERBOSE
            "${package_name} requested usage of a proprietary_binary but none exist for ${CMAKE_SYSTEM_PROCESSOR}"
    )
    return()
  endif()

  if(NOT DEFINED rapids-cmake-version)
    include("${rapids-cmake-dir}/rapids-version.cmake")
  endif()

  # Evaluate any magic placeholders in the proprietary_binary value including the
  # `rapids-cmake-version` value
  cmake_language(EVAL CODE "set(proprietary_binary ${proprietary_binary})")

  if(proprietary_binary)
    # download and extract the binaries since they don't exist on the machine
    include(FetchContent)
    set(pkg_name "${package_name}_proprietary_binary")

    # Prefer to use the download time for timestamp, instead of the timestamp in the archive unless
    # explicitly set by user. This allows for proper rebuilds when a projects url changes
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
      set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
    endif()

    FetchContent_Declare(${pkg_name} URL ${proprietary_binary})
    FetchContent_MakeAvailable(${pkg_name})

    # Tell the subsequent rapids_cpm_find where to search so that it uses this binary
    set(${package_name}_ROOT "${${pkg_name}_SOURCE_DIR}" PARENT_SCOPE)
    set(${package_name}_proprietary_binary ON PARENT_SCOPE)
  endif()
endfunction()
