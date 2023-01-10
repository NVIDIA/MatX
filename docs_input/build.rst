.. _building:

Building MatX
=============

As MatX is a header-only library, using it in your own projects is as simple as including only the core ``matx.h`` file. 
The ``matx.h`` include file gives full functionality of all MatX features, without requiring additional includes or API 
dependencies that the user must navigate. All classes required to use the MatX API is available from the ``matx.h`` include;
This includes arithmetic expressions, and all libraries included with CUDA (cuFFT, cuBLAS, CUB, cuSolver, cuRAND). 
Optional features of MatX that require downloading separate libraries use additional include files to
be explicit about their requirements.

The MatX CMake build configuration is intented to help download any libraries for both the required and optional features.
The CPM_ build system is used to help with package management and version control. By default, CPM will fetch other packages
from the internet. Alternatively, the option ``CPM_USE_LOCAL_PACKAGES`` can be used to point to local downloads in an air-gapped
or offline environment. Choosing local versions of packages uses the typical ``find_packages`` CMake search methods. Please see 
the CPM_ documentation or the documentation for each package for more information.


System Requirements
-------------------
MatX requires **CUDA 11.4** or higher, and **g++ 9.3** or higher for the host compiler. Clang may work as well, but it's currently 
untested. Other requirements for optional components are listed below.

.. warning:: Using MatX with an unsupported compiler may result in compiler and/or runtime errors.

Required Third-party Dependencies
---------------------------------

- CPM_ (this is included in the project source, so does not require a separate download)
- `fmt <https://github.com/fmtlib/fmt>`_ (CPM dependency)
- `nloghmann::json <https://github.com/nlohmann/json>`_ (CPM dependency)
- `rapids-cmake <https://github.com/rapidsai/rapids-cmake>`_
- `libcudacxx <https://github.com/NVIDIA/libcudacxx>`_

.. _CPM: https://github.com/cpm-cmake/CPM.cmake 


Optional Third-party Dependencies
---------------------------------
- `GoogleTest <https://github.com/google/googletest>`_
- `pybind11 <https://github.com/pybind/pybind11>`_
- `nvbench <https://github.com/NVIDIA/nvbench>`_
- `cutensor <https://developer.nvidia.com/cutensor>`_
- `cutensornet <https://docs.nvidia.com/cuda/cuquantum/cutensornet>`_

Build Options
=============
MatX provides 5 primary options for builds, and each can be configured independently:

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
  * - NVTX Flags
    - ``-DMATX_NVTX_FLAGS=ON``    


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

Building documentation has a separate list of requirements from all other build types. MatX requires the following packages to build
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

MatX Library Linking
====================
MatX defaults to Hidden Visibility due to compile requirements from pybind (https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes). 
Hidden Visibility hides symbols from the C++ linker, and will prevent a user from accessing functions from other translation units. If inheriting the MatX Build system, this will also prevent user-space symbols from being enabled, which may be a problem for multi-library or resource projects intended 
to be linked by later users. Visibility settings can be changed in the user's build environment, or specific symbols can be enabled through the C++ visibility support features (https://gcc.gnu.org/wiki/Visibility).


MatX in Offline Environments
============================
The MatX build system and CPM provide an easy-to-use mechanism to build projects using MatX in computing environments that do not have access to the internet. 
As described earlier, CPM provides a convenient mechanism to identify and locally cache all of the required third-party dependencies, which can 
then be packaged and delivered to offline systems manually. It is easy to package a build of MatX in preparation of deployment to closed area, all you need is 
an internet-enabled computer to prepare your package. The steps below outline the process for preparing your package, compressing it for transfer to your system,
and building on the offline system.

- Clone the MatX repository on an internet-enabled environment (this does not need to be identical to the deployment environment, but is simpler if it is / can build MatX)

  .. code-block:: shell

    git clone git@github.com:NVIDIA/MatX.git


- Determine the location you would like to build the CPM cache at, and export the variable.

  .. code-block:: shell

    export CPM_SOURCE_CACHE $HOME_ONLINE/matx_cpm_cache
    
- Build MatX with the build options required by your project, following the steps outlined above

- TAR and Compress the CPM cache for easy transport

  .. code-block:: shell

    tar -czvf matx_cache_VERS_NUM_.tar.gz $HOME_ONLINE/matx_cpm_cache
    
- Transfer MatX Source code and CPM cache to your offline system 

- Uncompress your cache TAR in a location available while building MatX

  .. code-block:: shell

    tar -xvf matx_cache_VERS_NUM_.tar.gz  $HOME_OFFLINE
    
- Export the CPM_SOURCE_CACHE to your environment before building MatX

  .. code-block:: shell

    export CPM_SOURCE_CACHE $HOME_OFFLINE/matx_cpm_cache

    
- Build your MatX project per your standard process, CPM will automatically use the cache



