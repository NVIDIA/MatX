# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
get_override_json
--------------------------

. code-block:: cmake

  get_override_json(package_name output_variable)

#]=======================================================================]
function(get_override_json package_name output_variable)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.get_override_json")
  string(TOLOWER "${package_name}" package_name)
  get_property(json_data GLOBAL PROPERTY rapids_cpm_${package_name}_override_json)
  set(${output_variable} "${json_data}" PARENT_SCOPE)
endfunction()
