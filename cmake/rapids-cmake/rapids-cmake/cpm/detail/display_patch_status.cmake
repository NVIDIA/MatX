# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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

    set(err_file "${CMAKE_BINARY_DIR}/rapids-cmake/patches/${package_name}/err")
    if(EXISTS "${err_file}")
      file(READ "${err_file}" contents)
      message(FATAL_ERROR "${contents}")
    endif()

  endif()
endfunction()
