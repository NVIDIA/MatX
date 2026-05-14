RAPIDS-CMake Basics
###################


Installation
************

The ``rapids-cmake`` module is designed to be acquired at configure time in your project.
Put the ``RAPIDS.cmake`` script, which handles fetching the rest of the module's content
via CMake's `FetchContent <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_,
into your repository.

.. code-block:: cmake

  cmake_minimum_required(...)

  include(${CMAKE_CURRENT_LIST_DIR}/RAPIDS.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  project(...)

Usage
*****

``rapids-cmake`` is designed for projects to use only the subset of features that they need. To enable
this ``rapids-cmake`` comprises the following primary components:

- :ref:`cmake <common>`
- :ref:`cpm <cpm>`
- :ref:`cython <cython>`
- :ref:`cuda <cuda>`
- :ref:`export <export>`
- :ref:`find <find>`
- :ref:`testing <testing>`

There are two ways projects can use ``rapids-cmake`` functions.

1. Call ``include(rapids-<component>)``, which imports commonly used functions for the component.
2. Load each function independently via ``include(${rapids-cmake-dir}/<component>/<function_name>.cmake)``.

Overriding RAPIDS.cmake
***********************

At times projects or developers will need to verify ``rapids-cmake`` branches. To do this you can set variables that control which repository ``RAPIDS.cmake`` downloads, which should be done like this:

.. code-block:: cmake

  # To set the version that is pulled (this must be set for RAPIDS.cmake to work):
  set(rapids-cmake-version "<version>")

  # To override the GitHub repository:
  set(rapids-cmake-repo "<my_fork>")

  # To use an exact Git SHA:
  set(rapids-cmake-sha "<my_git_sha>")

  # To use a Git tag:
  set(rapids-cmake-tag "<my_git_tag>")

  # To override the repository branch:
  set(rapids-cmake-branch "<my_feature_branch>")

  # Or to override the entire repository URL (e.g. to use a GitLab repo):
  set(rapids-cmake-url "https://gitlab.com/<my_user>/<my_fork>/-/archive/<my_branch>/<my_fork>-<my_branch>.zip")

  # To override the usage of fetching the repository without git info
  # This only works when specifying
  #
  # set(rapids-cmake-fetch-via-git "ON")
  # set(rapids-cmake-branch "release/<cal_ver>")
  #
  # or
  # set(rapids-cmake-fetch-via-git "ON")
  # set(rapids-cmake-url "https://gitlab.com/<my_user>/<private_fork>/")
  # set(rapids-cmake-sha "ABC123")
  #
  set(rapids-cmake-fetch-via-git "ON")

  include(${CMAKE_CURRENT_LIST_DIR}/RAPIDS.cmake)

A few notes:

- An explicitly defined ``rapids-cmake-url`` will always be used
- ``rapids-cmake-sha`` takes precedence over ``rapids-cmake-tag``
- ``rapids-cmake-tag`` takes precedence over ``rapids-cmake-branch``
- The CMake variable ``rapids-cmake-version`` must be set to a rapids-cmake version, formatted as ``MAJOR.MINOR``
- ``RAPIDS.cmake`` should be copied in and placed next to the above file
