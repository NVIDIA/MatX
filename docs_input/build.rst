.. _building:

Building MatX
=============

As MatX is a header-only library, using it in your own projects is as simple as including only the core ``matx.h`` file. 
The ``matx.h`` include file is intended to give full functionality of all MatX features that do not require downloading or
including separate libraries. This includes arithmetic expressions, and all libraries included with CUDA (cuFFT, cuBLAS, 
CUB, cuSolver, cuRAND). Optional features of MatX that require downloading separate libraries use additional include files to
be explicit about their requirements.

The MatX CMake build configuration is intented to help download any libraries for both the required and optional features.
The CPM_ build system is used to help with package management and version control. By default, CPM will fetch other packages
from the internet. Alternatively, the option ``CPM_USE_LOCAL_PACKAGES`` can be used to point to local downloads in an air-gapped
or offline environment. Choosing local versions of packages uses the typical ``find_packages`` CMake search methods. Please see 
the CPM_ documentation or the documentation for each package for more information.

.. _CPM: https://github.com/cpm-cmake/CPM.cmake
.. _GoogleTest: https://github.com/google/googletest
.. _pybind11: https://github.com/pybind/pybind11

Core Requirements
-----------------
MatX requires **CUDA 11.4** or higher, and **g++ 9.3** or higher for the host compiler. Clang may work as well, but it's currently 
untested. Other requirements for optional components are listed below.

.. warning:: Using MatX will an unsupported compiler may result in compiler and/or runtime errors.

Build Choices
=============
MatX provides 4 different types of build types, and can each be configured independently:

.. list-table::
  :widths: 60 40
  :header-rows: 1

  * - Type
    - CMake Option
  * - Unit Tests
    - ``-DMATX_BUILD_TESTS=ON`` 
  * - Benchmarks
    - ``-DMATX_BUILD_BENCHMARKS=ON`` 
  * - Examples
    - ``-DMATX_BUILD_EXAMPLES=ON`` 
  * - Documentation
    - ``-DMATX_BUILD_DOCS=ON``             


Everything but documentation requires building MatX source code, and all requirements in the first section of this document apply.
Building documentation will be covered later in this document.

Unit tests
----------
MatX unit tests are compiled using **Google Test** as the test framework, and **pybind11** for verification. pybind11 provides an interface
to compare common Numpy and Scipy results with MatX without reimplementing complex functionality in C++. Required versions for these 
libraries are:

**Google test**: 1.11+

**pybind11**: 2.6.2

Both Google Test and pybind11 will be automatically downloaded by CPM when unit tests are enabled. If an offline copy of them exists, 
``CPM_USE_LOCAL_PACKAGES`` can be used to override the download. 

To build unit tests, pass the argument ``-DMATX_BUILD_TESTS=ON`` to CMake to configure the build environment, then issue:

.. code-block:: shell

    make -j test

This will compile and run all unit tests. For more control over which tests to run, you may run test/matx_test directly with parameters 
defined by Google Test (https://github.com/google/googletest). To run matx_test directly, you must be inside the build/test directory 
for the correct paths to be set. For example, to run only tests with the name FFT:

.. code-block:: shell

    test/matx_test --gtest_filter="*FFT*"

Examples
--------

MatX provides several example applications that show different capabilities of MatX. When the ``-DMATX_BUILD_EXAMPLES=ON`` CMake argument
is specified the ``build/examples`` directory will contain a separate binary file for each example. Each example can be run by simply
executing the binary.


Benchmarks
----------
MatX uses the NVBench software for the benchmarking framework. Like other packages, NVBench will be download using CPM according to
the methods mentioned above.

NVBench has a small library that will be compiled on the first `make` run. Benchmarks can be run using the ``bench/matx_bench`` executable,
and all options to filter or modify benchmark runs can be found in the nvbench_ project documentation.

.. _nvbench: https://github.com/NVIDIA/nvbench


Documentation
-------------

Building documentation has a separete list of requirements from all other build types. MatX requires the following packages to build
documentation:

**Breate**: 4.31.0

**Doxygen**: 1.9.1

**Sphinx**: 4.3.1

**sphinx-book-theme**: 0.1.7

**libjs-mathjax**

**texlive-font-utils**

Building documentation must be done separately from other build options as to minimize the requirements needed. After configuring CMake with
``-DMATX_BUILD_DOCS=ON`` and typing ``make``, Doxygen, Sphinx, and Breathe will parse the source to build the documentation. Once complete, a 
directory ``build/docs_input/sphinx`` will be created containing all documentation files, and an ``index.html`` entry point that can be used
to browse the documentation. Note that the most recent version of the documentation is also hosted at:

https://nvidia.github.io/MatX/

