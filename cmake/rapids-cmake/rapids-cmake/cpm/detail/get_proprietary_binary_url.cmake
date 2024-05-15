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
rapids_cpm_get_proprietary_binary_url
-------------------

.. versionadded:: v23.04.00

Generated the url for the associated proprietary binary for the given project based
on the current CPU target architecture ( x86_64, aarch64, etc )

 .. note::
  if override => the proprietary entry only in the override will be evaluated
  if no override => the proprietary entry only in the default will be evaluated

#]=======================================================================]
function(rapids_cpm_get_proprietary_binary_url package_name version url_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_get_proprietary_binary_url")

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

  # Determine the CUDA Toolkit version so that we properly evaluate the placeholders in
  # `proprietary_binary`
  if(proprietary_binary MATCHES "{cuda-toolkit-version")
    find_package(CUDAToolkit REQUIRED)
    set(cuda-toolkit-version ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
    set(cuda-toolkit-version-major ${CUDAToolkit_VERSION_MAJOR})
  endif()

  # Evaluate any magic placeholders in the proprietary_binary value including the
  # `rapids-cmake-version` value
  cmake_language(EVAL CODE "set(proprietary_binary ${proprietary_binary})")

  # Tell the caller what the URL will be for this binary
  set(${url_var} "${proprietary_binary}" PARENT_SCOPE)
endfunction()
