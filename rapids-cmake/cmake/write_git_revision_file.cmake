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
rapids_cmake_write_git_revision_file
------------------------------------

.. versionadded:: v21.10.00

Generate a C++ header file that holds git revision information of the calling project.

.. code-block:: cmake

  rapids_cmake_write_git_revision_file(<target_name> file_path [PREFIX <prefix>])

Creates a global interface target called `target_name` that holds the includes to
the generated header with the macros for git branch, sha1, version, and if any uncommited
changes exist. Users of the header file must use
:cmake:command:`target_link_libraries <cmake:command:target_link_libraries>` to the target
so that the header is generated before it is used.

``PREFIX``
    Prefix for all the C++ macros. By default if not explicitly specified it will be equal
    to the projects name ( CMake variable `PROJECT_NAME` ).

This information will be recorded in the following defines:

  - <PREFIX>_GIT_BRANCH
    Will store the current git branch name, otherwise when in a detached HEAD state will
    store `HEAD`.

  - <PREFIX>_GIT_SHA1
    Will store the full SHA1 for the current git commit if one exists.

  - <PREFIX>_GIT_IS_DIRTY
    Will exist if any git tracked file has any modifications that aren't commited ( dirty ).

  - <PREFIX>_GIT_VERSION
    Will store `<tag>[-<distance>-g<sha1>[-dirty]]` computed from running
    `git describe --tags --dirty --always`. For example "v21.10.00a-18-g7efb04f-dirty" indicates
    that the lastest commit is "7efb04f" but has uncommitted changes (`-dirty`), and
    that we are "18" commits after tag "v21.10.00a".

``file_path``
    Either an absolute or relative path.
    When a relative path, the absolute location will be computed from
    :cmake:variable:`CMAKE_CURRENT_BINARY_DIR <cmake:variable:CMAKE_CURRENT_BINARY_DIR>`

.. note::
  If `git` doesn't exist or the project doesn't use `git`, the header
  will still be written. The branch, sha1, and version defines will be set to
  `unknown` and the project won't be considered dirty.


Result Targets
^^^^^^^^^^^^^^^^
  `target_name` target will be created. Consuming libraries/executables
   of the generated header must use the target via
   :cmake:command:`target_link_libraries <cmake:command:target_link_libraries>`
   for correct builds.

#]=======================================================================]
function(rapids_cmake_write_git_revision_file target file_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.write_git_revision_file")

  set(options "")
  set(one_value PREFIX)
  set(multi_value "")
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  cmake_path(IS_RELATIVE file_path is_relative)
  if(is_relative)
    cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR ${file_path} OUTPUT_VARIABLE output_path)
  else()
    set(output_path "${file_path}")
  endif()

  if(NOT _RAPIDS_PREFIX)
    set(_RAPIDS_PREFIX "${PROJECT_NAME}")
  endif()

  # Find Git
  find_package(Git QUIET)

  add_custom_target(${target}_compute_git_info ALL
                    BYPRODUCTS "${file_path}"
                    COMMENT "Generate git revision file for ${target}"
                    COMMAND ${CMAKE_COMMAND} -DWORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
                            -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                            -D_RAPIDS_GIT_PREFIX=${_RAPIDS_PREFIX}
                            -DTEMPLATE_FILE=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/git_revision.hpp.in
                            -DFILE_TO_WRITE=${file_path} -P
                            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/compute_git_info.cmake
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Generate a target that depends on compute_git_info This is what other targets will use to get
  # the build path and makes sure that we have correct parallel builds
  add_library(${target} INTERFACE)
  add_dependencies(${target} ${target}_compute_git_info)

  cmake_path(GET file_path PARENT_PATH file_path_dir)
  target_include_directories(${target} INTERFACE ${file_path_dir})

endfunction()
