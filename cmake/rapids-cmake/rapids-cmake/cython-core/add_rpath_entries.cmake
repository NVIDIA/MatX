# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cython_add_rpath_entries
-------------------------------
.. versionadded:: v24.02.00

Set the RPATH entries for all targets associated with a provided associated target.

.. code-block:: cmake

  rapids_cython_add_rpath_entries(
      TARGET <associated_target>
      PATHS <path1> <path2> ...
      [ROOT_DIRECTORY <root-dir>]
  )

This function will affect all targets created up to the point of this call. It
will have no effect on targets created afterwards.

``TARGET``
  The associated target for which we are setting RPATH entries. Any target
  created using :cmake:command:`rapids_cython_create_modules` with the argument
  `ASSOCIATED_TARGET associated_target` will have its RPATH entries updated.

``PATHS``
  The paths to add to the RPATH. Paths may either be absolute or relative to
  the ROOT_DIRECTORY. The paths are always converted to be relative to the
  current directory i.e relative to $ORIGIN in the RPATH.

``ROOT_DIRECTORY``
  The ROOT_DIRECTORY for the provided paths. Defaults to ${PROJECT_SOURCE_DIR}.
  Has no effect on absolute paths. If the ROOT_DIRECTORY is a relative path, it
  is assumed to be relative to the directory from which
  `rapids_cython_add_rpath_entries` is called.

#]=======================================================================]
function(rapids_cython_add_rpath_entries)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cython.add_rpath_entries")

  set(options)
  set(one_value ROOT_DIRECTORY TARGET)
  set(multi_value PATHS)
  cmake_parse_arguments(_RAPIDS_CYTHON "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # By default paths are relative to the current project root.
  if(NOT _RAPIDS_CYTHON_ROOT_DIRECTORY)
    set(_RAPIDS_CYTHON_ROOT_DIRECTORY "${PROJECT_SOURCE_DIR}")
  endif()

  # Transform all paths to paths relative to the current directory.
  set(cleaned_paths)
  cmake_path(ABSOLUTE_PATH _RAPIDS_CYTHON_ROOT_DIRECTORY)
  foreach(path IN LISTS _RAPIDS_CYTHON_PATHS)
    if(NOT IS_ABSOLUTE path)
      cmake_path(ABSOLUTE_PATH path BASE_DIRECTORY "${_RAPIDS_CYTHON_ROOT_DIRECTORY}")
    endif()
    list(APPEND cleaned_paths "${path}")
  endforeach()

  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(platform_rpath_origin "@loader_path")
  else()
    set(platform_rpath_origin "$ORIGIN")
  endif()
  get_property(targets GLOBAL PROPERTY "rapids_cython_associations_${_RAPIDS_CYTHON_TARGET}")
  foreach(target IN LISTS targets)
    # Compute the path relative to the current target.
    set(target_paths)
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    foreach(target_path IN LISTS cleaned_paths)
      cmake_path(RELATIVE_PATH target_path BASE_DIRECTORY "${target_source_dir}")
      list(APPEND target_paths "${platform_rpath_origin}/${target_path}")
    endforeach()
    list(JOIN target_paths ";" target_paths)

    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${target_paths}")
  endforeach()
endfunction()
