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
rapids_cpm_display_patch_status
-------------------------------

.. versionadded:: v22.10.00

Displays the result of any patches applied to the requested package

.. code-block:: cmake

  rapids_cpm_display_patch_status(<pkg>)

#]=======================================================================]
function(rapids_cpm_display_patch_status package_name)
  # Only display the status information on the first execution of the call
  if(${package_name}_ADDED)
    list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.display_patch_status")
    set(log_file "${CMAKE_BINARY_DIR}/rapids-cmake/patches/${package_name}/log")
    if(EXISTS "${log_file}")
      file(STRINGS "${log_file}" contents)
      foreach(line IN LISTS contents)
        message(STATUS "${line}")
      endforeach()
    endif()
  endif()
endfunction()
