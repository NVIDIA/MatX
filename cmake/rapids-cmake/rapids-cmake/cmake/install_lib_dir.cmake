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
rapids_cmake_install_lib_dir
------------------------------

.. versionadded:: v21.10.00

Establish a variable that holds the library installation directory.

  .. code-block:: cmake

    rapids_cmake_install_lib_dir( out_variable_name [MODIFY_INSTALL_LIBDIR] )

Establishes a variable that holds the correct library installation directory
( lib or lib64 or lib/<multiarch-tuple> ). This function is CONDA aware and
will return `lib` when it detects a project is installing in the CONDA_PREFIX

Also offers the ability to modify :cmake:command:`CMAKE_INSTALL_LIBDIR <cmake:command:install>` to
be the computed installation directory.


Result Variables
^^^^^^^^^^^^^^^^
  :cmake:command:`CMAKE_INSTALL_LIBDIR <cmake:command:install>` will be modifed to be the computed relative directory
  (lib or lib64 or lib/<multiarch-tuple>) when `MODIFY_INSTALL_LIBDIR` is provided

#]=======================================================================]
function(rapids_cmake_install_lib_dir out_variable_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cmake.install_lib_dir")

  set(modify_install_libdir FALSE)
  if(ARGV1 STREQUAL "MODIFY_INSTALL_LIBDIR")
    set(modify_install_libdir TRUE)
  endif()

  set(install_prefix "${CMAKE_INSTALL_PREFIX}")
  cmake_path(ABSOLUTE_PATH install_prefix NORMALIZE)

  set(use_conda_lib_dir FALSE)
  if(CMAKE_VERSION VERSION_LESS 3.22)
    # Starting with 3.22, CMake is fully aware of the conda 'lib' requirements so we don't need to
    # do this check at all
    if(DEFINED ENV{CONDA_BUILD} AND DEFINED ENV{PREFIX})
      set(conda_prefix "$ENV{PREFIX}")
      cmake_path(ABSOLUTE_PATH conda_prefix NORMALIZE)
      if(install_prefix STREQUAL conda_prefix)
        set(use_conda_lib_dir TRUE)
      endif()
    elseif(DEFINED ENV{CONDA_PREFIX})
      set(conda_prefix "$ENV{CONDA_PREFIX}")
      cmake_path(ABSOLUTE_PATH conda_prefix NORMALIZE)
      if(install_prefix STREQUAL conda_prefix)
        set(use_conda_lib_dir TRUE)
      endif()
    endif()
  endif()

  set(computed_path)
  if(use_conda_lib_dir)
    # CONDA requires everything to be installed to 'lib' no matter the distro
    set(computed_path "lib")
    if(modify_install_libdir)
      # GNUInstallDirs sets `CMAKE_INSTALL_LIBDIR` as a cache path, so we need to do that as well
      set(CMAKE_INSTALL_LIBDIR ${computed_path} CACHE PATH
                                                      "Object code libraries (${computed_path})")

      # Make sure our path overrides any local variable
      set(CMAKE_INSTALL_LIBDIR ${computed_path} PARENT_SCOPE)
    endif()
  else()
    # We need to defer to GNUInstallDirs but not allow it to set CMAKE_INSTALL_LIBDIR
    set(remove_install_dir TRUE)
    if(DEFINED CMAKE_INSTALL_LIBDIR)
      set(remove_install_dir FALSE)
    endif()

    include(GNUInstallDirs)
    set(computed_path "${CMAKE_INSTALL_LIBDIR}")
    if(modify_install_libdir)
      # GNUInstallDirs will have set `CMAKE_INSTALL_LIBDIR` as a cache path So we only need to make
      # sure our path overrides any local variable
      set(CMAKE_INSTALL_LIBDIR ${computed_path} PARENT_SCOPE)
    endif()

    if(remove_install_dir)
      unset(CMAKE_INSTALL_LIBDIR CACHE)
    endif()
  endif()

  set(${out_variable_name} ${computed_path} PARENT_SCOPE)

endfunction()
