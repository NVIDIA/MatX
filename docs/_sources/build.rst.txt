.. _building:

Building MatX
======================================================

Requirements
------------
- CUDA 11.4
- g++ 9.0 or above

Unit tests
**********
pybind11 source build and installed
Python3 includes (python3-dev package)

Multi GPU
*********
NvSHMEM

Building
--------
While the MatX library does not require compiling, building unit tests, benchmarks, or examples must be compiled.
To build all components, issue the standard cmake build commands in a cloned repo:

.. code-block:: shell

    mkdir build && cd build
    cmake ..
    make -j

Unit tests and examples are built by default. Documentation building is disabled by default. To disable specific builds, 
use the cmake properties with the OFF flag:
BUILD_EXAMPLES
BUILD_TESTS
For example, to disable unit test building:

.. code-block:: shell

    mkdir build && cd build
    cmake -DBUILD_TESTS=OFF ..
    make -j

To build documentation use the cmake flag ``-DBUILD_DOCS=ON``


Unit tests
----------
MatX contains a suite of unit tests to test functionality of the primitive functions, plus other signal processing pipelines.
Due to the nature of certain unit tests, the amount of data used as input and output for comparison can get quite large, and
as a result, test vectors are generated dynamically using Python. Version 2.6.2 of the pybind11 library is required to be built
and installed before building the MatX unit tests. Once built, the MatX build system will try to find pybind11 in common installation
directories. If it cannot be found, it may be specified manually by passing the pybind11_DIR CMake parameter:

.. code-block:: shell

    cmake -Dpybind11_DIR=/matx_libs/pybind11
    

To run the unit tests, from the cmake build directory run:

.. code-block:: shell

    make -j test

This will execute all unit tests defined. If you wish to execute a subset of tests, or run with different options, you
may run test/matx_test directly with parameters defined by Google Test (https://github.com/google/googletest). To run matx_test
directly, you must be inside the build/test directory for the correct paths to be set. For example,
to run only tests with the name FFT:

.. code-block:: shell

    cd build/test
    ./matx_test --gtest_filter="*FFT*"


Benchmarks
----------
MatX uses the NVBench software for the benchmarking framework. To run the benchmarks, you must have NVBench downloaded from 
https://github.com/NVIDIA/nvbench. When specifying the CMake parameters benchmarks can be enabled by:

.. code-block:: shell

    cmake .. -DNVBENCH_DIR=/my/nvbench/dir -DBUILD_BENCHMARKS=ON

NVBench has a small library that will be compiled on the first `make` run. A binary called `matx_bench` will be produced
inside of build/bench, and all parameters to the binary can be found in the NVBench documentation.


Multi-GPU Support
-----------------
MatX examples and unit tests can be compiled with multi-GPU support using the NVSHMEM library. NVSHMEM must be compiled and/or
installed before building MatX with multi-GPU support. The build system will look in common locations for NVSHMEM libraries and
include files, but the installation directory can be overridden on the cmake configuration:

.. code-block:: shell

    cmake -DMULTI_GPU=ON -DNVSHMEM_DIR=/usr/local/nvshmem ..

If nvshmem is detected properly, you will see a message similar to the following during cmake:

``-- Found Nvshmem: /usr/local/nvshmem/lib/libnvshmem.a``

