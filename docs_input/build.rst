.. _building:

Build Integration
=================

As MatX is a header-only library, using it in your own projects is as simple as including only the core ``matx.h`` file. 
The ``matx.h`` include file gives full functionality of all MatX features, without requiring additional includes or API 
dependencies that the user must navigate. All classes required to use the MatX API is available from the ``matx.h`` include;
This includes arithmetic expressions, and all libraries included with CUDA (cuFFT, cuBLAS, CUB, cuSolver, cuRAND). 
Optional features of MatX that require downloading separate libraries use additional include files to
be explicit about their requirements.

The MatX CMake build configuration is intented to help download any libraries for both the required and optional features.
The CPM build system is used to help with package management and version control. By default, CPM will fetch other packages
from the internet. Alternatively, the option ``CPM_USE_LOCAL_PACKAGES`` can be used to point to local downloads in an air-gapped
or offline environment. Choosing local versions of packages uses the typical ``find_packages`` CMake search methods. Please see 
the CPM documentation or the documentation for each package for more information.


System Requirements
-------------------
MatX requires **CUDA 11.8** or **CUDA 12.2.1** or higher, and **g++ 9.3+**, **clang 17+**, or **nvc++ 24.5** for the host compiler. See the CUDA toolkit documentation
for supported host compilers. Other requirements for optional components are listed below.

.. warning:: Using MatX with an unsupported compiler may result in compiler and/or runtime errors.

Required Third-party Dependencies
---------------------------------

- `CCCL <https://github.com/NVIDIA/cccl>`_ 3.0.0 or higher


Optional Third-party Dependencies
---------------------------------
- `CMake <https://cmake.org/>`_ 3.23.1+ (Required for running unit tests, benchmarks, or examples)
- `GoogleTest <https://github.com/google/googletest>`_ 1.11.0+ (Required to run unit tests)
- `pybind11 <https://github.com/pybind/pybind11>`_ 2.6.2+ (Required for file I/O and some unit tests)
- `nvbench <https://github.com/NVIDIA/nvbench>`_ Commit 1a13a2e (Required to run benchmarks)
- `cutensor <https://developer.nvidia.com/cutensor>`_ 2.0.1.2+ (Required when using `einsum`)
- `cutensornet <https://docs.nvidia.com/cuda/cuquantum/cutensornet>`_ 24.03.0.4+ (Required when using `einsum`)
- `cuDSS <https://developer.nvidia.com/cudss>`_ 0.4.0.2+ (Required when using `solve` on sparse matrices)

Host (CPU) Support
------------------
Host support is provided by the C++ standard library and different CPU math libraries. NVIDIA's NVPL_ library can
be used for FFT, BLAS, and LAPACK support on ARM. Other supported libraries include FFTW_ for FFT support, OpenBLAS_ or BLIS_
for BLAS support, and OpenBLAS for LAPACK support. BLAS enables matrix and vector product functions like ``matmul``, ``outer``,
and ``matvec``. LAPACK enables all matrix decomposition/factorization functions like ``chol``, ``qr``, ``svd``, and others.

Below are the CMake options to enable each library:

* **NVPL** support for FFT, BLAS, and LAPACK operations on ARM: ``-DMATX_EN_NVPL=ON``
* **FFTW** support for FFT: ``-DMATX_EN_X86_FFTW=ON``
* **BLIS** support for BLAS operations: ``-DMATX_EN_BLIS=ON``
* **OpenBLAS** support for BLAS and LAPACK operations: ``-DMATX_EN_OPENBLAS=ON``

**Note**: Enabling NVPL will enable NVPL libraries for FFT, BLAS and LAPACK, and so it cannot be used in conjunction with other CPU libraries
at this time. The ``blas_DIR`` CMake variable may be needed to configure either BLIS or OpenBLAS library

Currently all elementwise operators, reductions, and FFT/BLAS/LAPACK transforms are supported. Most host functions with
the exception of reductions support multithreading. A more detailed breakdown of the Host support/limitations for each function
is available in the :ref:`Executor Compatibility <executor_compatibility>` section.

.. _NVPL: https://developer.nvidia.com/nvpl
.. _OpenBLAS: https://www.openblas.net/
.. _FFTW: http://www.fftw.org/
.. _BLIS: https://github.com/flame/blis

Build Targets
=============
MatX provides 4 primary targets for builds, and each can be configured independently:

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
to compare common Numpy and Scipy results with MatX without reimplementing complex functionality in C++. 

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

**Doxygen**: 1.11.0

**Sphinx**: 4.3.1

**sphinx-book-theme**: 0.1.7

**libjs-mathjax**

**texlive-font-utils**

Building documentation must be done separately from other build options as to minimize the requirements needed. After configuring CMake with
``-DMATX_BUILD_DOCS=ON`` and typing ``make``, Doxygen, Sphinx, and Breathe will parse the source to build the documentation. Once complete, a 
directory ``build/docs_input/sphinx`` will be created containing all documentation files, and an ``index.html`` entry point that can be used
to browse the documentation. Note that the most recent version of the documentation is also hosted at:

https://nvidia.github.io/MatX/

Additional Build Options
========================

There are several additional build options to control code generation or other instrumentation.
By default, all of these options are OFF.

.. list-table::
  :widths: 60 40
  :header-rows: 1

  * - Type
    - CMake Option
  * - NVTX Flags
    - ``-DMATX_NVTX_FLAGS=ON``
  * - 32-bit Indices
    - ``-DMATX_BUILD_32_BIT=ON``
  * - File I/O Support
    - ``-DMATX_EN_FILEIO=ON``
  * - Code Coverage
    - ``-DMATX_EN_COVERAGE=ON``
  * - Complex Operations NaN/Inf Handling
    - ``-DMATX_EN_COMPLEX_OP_NAN_CHECKS=ON``
  * - CUDA Line Info
    - ``-DMATX_EN_CUDA_LINEINFO=ON``

NVTX Flags
----------

Enabling NVTX flags adds NVTX ranges for existing MatX operations and enables users to add NVTX ranges to their own code
using the provided MatX NVTX macros. See :ref:`nvtx-profiling` for more information.

32-bit Indices
--------------

Enabling 32-bit indices utilizes 32-bit signed integers as the ``index_t`` data type in MatX.
This data type is used for sizing tensors and for indexing into tensors. By default, ``index_t``
is a 64-bit signed integer type, which allows for tensors exceeding 2\ :sup:`31`-1 in size.
If all of the tensors in your application will be less than 2\ :sup:`31`-1 elements, then using
a 32-bit index type improves performance for some operations.

File I/O Support
----------------

Enables support to read and write data from/to csv, npy, and mat files.
This option adds pybind11 as a dependency. See :ref:`io` for more information on the available I/O functions.

Code Coverage
-------------

Adds compiler and linker arguments (e.g., ``-fprofile-arcs -ftest-coverage``) to support use of code coverage tools like gcov.

Complex Operations NaN/Inf Handling
-----------------------------------

Complex multiplication using the ``cuda::std::complex<T>`` types can be directly implemented via:

.. code-block:: cpp

  const cuda::std::complex<float> prod(
    x.real() * y.real() - x.imag() * y.imag(),
    x.real() * y.imag() + x.imag() * y.real());

With this implementation, typical propagation of NaNs and infinite values will apply. For example, the
following direct complex multiplication implementation

.. code-block:: cpp

  const cuda::std::complex<float> x(
    cuda::std::numeric_limits<float>::infinity(), cuda::std::numeric_limits<float>::infinity());
  const cuda::std::complex<float> y(std::nan(""), 1.0f);

will yield ``NaN`` for both the real and imaginary components.
Annex G of the C11 Standard introduces different handling for such cases so that, for example, the
above case yields positive or negative infinity in each of the components. The CCCL library used by MatX for
the ``cuda::std::complex`` implementation supports this extra handling for multiplication and division
(specifically, ``operator*()`` and ``operator/()``) of the complex type, but MatX disables it by default and
yields the semantics of the direct implementation above.
Using the ``-DMATX_EN_COMPLEX_OP_NAN_CHECKS=ON`` CMake option or otherwise defining the ``MATX_EN_COMPLEX_OP_NAN_CHECKS``
macro will enable these additional checks. Enabling this option introduces extra cost in complex multiplication and division.

CUDA Line Info
--------------

Using the ``-DMATX_EN_CUDA_LINEINFO=ON`` CMake command-line argument will enable CUDA kernel line information in the
produced libraries or executables. This is equivalent to using the ``-lineinfo`` option with NVCC. Line information is
useful when using the debugger or profiler (e.g., when using the ``--import-source=yes`` option with Nsight Compute).

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



