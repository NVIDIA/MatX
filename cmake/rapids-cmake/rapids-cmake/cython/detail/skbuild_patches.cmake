# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

include_guard(GLOBAL)

#[=======================================================================[.rst:
_set_python_extension_symbol_visibility
---------------------------------------

.. versionadded:: v22.06.00

The original version of this function in scikit-build runs a linker script to
modify the visibility of symbols. This version is a patch to avoid overwriting
the visibility of symbols because otherwise any library that exports symbols
with external linkage will have the visibility of those symbols changed
undesirably. We can remove this function once this issue is resolved in
scikit-build.

Issue: https://github.com/scikit-build/scikit-build/issues/668
PR: https://github.com/scikit-build/scikit-build/pull/703

#]=======================================================================]
function(_set_python_extension_symbol_visibility _target)
  include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/verify_init.cmake)
  rapids_cython_verify_init()

  if(PYTHON_VERSION_MAJOR VERSION_GREATER 2)
    set(_modinit_prefix "PyInit_")
  else()
    set(_modinit_prefix "init")
  endif()
  message("_modinit_prefix:${_modinit_prefix}")
  if("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set_target_properties(${_target} PROPERTIES LINK_FLAGS "/EXPORT:${_modinit_prefix}${_target}")
  elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(_script_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}-version-script.map)
    file(WRITE ${_script_path} # Note: The change is to this script, which does not indiscriminately
                               # mark all non PyInit symbols as local.
         "{global: ${_modinit_prefix}${_target}; };")
    set_property(TARGET ${_target} APPEND_STRING
                 PROPERTY LINK_FLAGS " -Wl,--version-script=\"${_script_path}\"")
  endif()
endfunction()
