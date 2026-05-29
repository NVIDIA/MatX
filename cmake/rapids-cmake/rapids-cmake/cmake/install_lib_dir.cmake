# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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
  :cmake:command:`CMAKE_INSTALL_LIBDIR <cmake:command:install>` will be modified to be the computed relative directory
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

  set(computed_path)

  # We need to defer to GNUInstallDirs but not allow it to set CMAKE_INSTALL_LIBDIR
  set(remove_install_dir TRUE)
  if(DEFINED CMAKE_INSTALL_LIBDIR)
    set(remove_install_dir FALSE)
  endif()

  include(GNUInstallDirs)
  set(computed_path "${CMAKE_INSTALL_LIBDIR}")
  if(modify_install_libdir)
    # GNUInstallDirs will have set `CMAKE_INSTALL_LIBDIR` as a cache path so we only need to make
    # sure our path overrides any local variable
    set(CMAKE_INSTALL_LIBDIR ${computed_path} PARENT_SCOPE)
  endif()

  if(remove_install_dir)
    unset(CMAKE_INSTALL_LIBDIR CACHE)
  endif()

  set(${out_variable_name} ${computed_path} PARENT_SCOPE)

endfunction()
