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
download_proprietary_binary
-------------------

.. versionadded:: v23.04.00

Download the associated proprietary binary from the providied URL and make
it part of the project with `FetchContent_MakeAvailable`

#]=======================================================================]
function(rapids_cpm_download_proprietary_binary package_name url)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rapids_cpm_download_proprietary_binary")

  # download and extract the binaries since they don't exist on the machine
  include(FetchContent)
  set(pkg_name "${package_name}_proprietary_binary")

  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
  endif()

  FetchContent_Declare(${pkg_name} URL ${url})
  FetchContent_MakeAvailable(${pkg_name})

  # Tell the subsequent rapids_cpm_find where to search so that it uses this binary
  set(${package_name}_ROOT "${${pkg_name}_SOURCE_DIR}" PARENT_SCOPE)
  set(${package_name}_proprietary_binary ON PARENT_SCOPE)
endfunction()
