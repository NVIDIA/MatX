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
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cmake_make_global
------------------------

.. versionadded:: v21.06.00

Make sure all provided targets have global visibility no matter how they
are constructed.

.. code-block:: cmake

    rapids_cmake_make_global(target_var)


CMake targets have visibility or scope where they can be referenced by name.
Any built-in target such as those created by :cmake:command:`add_library <cmake:command:add_library>` have
global visibility. Targets created with :cmake:command:`add_library(IMPORTED) <cmake:command:add_library>` by
default have directory visibility. This causes problems when trying to reason
about targets created by `CPM`, as they could be either of the above.

This function promotes the set of targets provided to have global visibility.
This makes it easier for users to reason about when/where they can reference
the targets.


``target_var``
    Holds the variable that lists all targets that should be promoted to
    GLOBAL scope

#]=======================================================================]
function(rapids_cmake_make_global target_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.make_global")
  foreach(target IN LISTS ${target_var})
    if(TARGET ${target})
      get_target_property(aliased_target ${target} ALIASED_TARGET)
      if(aliased_target)
        continue()
      endif()
      get_target_property(is_imported ${target} IMPORTED)
      get_target_property(already_global ${target} IMPORTED_GLOBAL)
      if(is_imported AND NOT already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
    endif()
  endforeach()
endfunction()
