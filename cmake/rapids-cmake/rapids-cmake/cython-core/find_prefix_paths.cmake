# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cython_find_prefix_paths
-------------------------------

.. versionadded:: v26.02.00

Find all paths that should be added to CMAKE_PREFIX_PATH according to Python entry points.

.. code-block:: cmake

  rapids_cython_find_prefix_paths(<python_executable> <paths_var>)

``python_executable``
  Path to the Python executable whose entry points to search

``paths_var``
  The variable to set with the resulting list of paths.

#]=======================================================================]
function(rapids_cython_find_prefix_paths python_executable paths_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cython.init")

  set(_get_entry_points
      [=[
import os
from importlib.metadata import entry_points
from importlib.resources import files

paths = []
for ep in entry_points(group="cmake.prefix"):
    p = files(ep.load())
    # Some entry_points expose a _paths attribute
    if hasattr(p, "_paths"):
        paths.extend(map(os.fspath, p._paths))
    else:
        paths.append(os.fspath(p))

print(";".join(paths))
]=])

  # Execute the Python at configure time and capture output
  execute_process(COMMAND ${python_executable} -c "${_get_entry_points}" OUTPUT_VARIABLE prefix_dirs
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(${paths_var} ${prefix_dirs} PARENT_SCOPE)

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endfunction()
