RAPIDS-CMake Basics
###################


Installation
************

The ``rapids-cmake`` module is designed to be acquired via CMake's `Fetch
Content <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_ into your project.

.. code-block:: cmake

  cmake_minimum_required(...)

  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.12/RAPIDS.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
  endif()
  include(${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  project(...)

Usage
*****

``rapids-cmake`` is designed for projects to use only the subset of features that they need. To enable
this `rapids-cmake` comprises the following primary components:

- `cmake <api.html#common>`__
- `cpm <api.html#cpm>`__
- `cython <api.html#cython>`__
- `cuda <api.html#cuda>`__
- `export <api.html#export>`__
- `find <api.html#find>`__
- `testing <api.html#testing>`__

There are two ways projects can use ``rapids-cmake`` functions.

1. Call ``include(rapids-<component>)``, which imports commonly used functions for the component.
2. Load each function independently via ``include(${rapids-cmake-dir}/<component>/<function_name>.cmake)``.

Overriding RAPIDS.cmake
***********************

At times projects or developers will need to verify ``rapids-cmake`` branches. To do this you need to override the default git repository and branch that ``RAPIDS.cmake`` downloads, which should be done
like this:

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
    rapids-cmake
    GIT_REPOSITORY https://github.com/<my_fork>/rapids-cmake.git
    GIT_TAG        <my_feature_branch>
  )
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.12/RAPIDS.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)


This tells ``FetchContent`` to ignore the explicit url and branch in ``RAPIDS.cmake`` and use the
ones provided.

An incorrect approach that people try is to modify the ``file(DOWNLOAD)`` line to point to the
custom ``rapids-cmake`` branch. That doesn't work as the downloaded ``RAPIDS.cmake`` contains
which version of the rapids-cmake repository to clone.
