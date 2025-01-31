Welcome To MatX's Documentation!
================================

**MatX** is a modern C++ library for numerical computing on NVIDIA GPUs and CPUs. The main
features include:

* Compile-time expression evaluation for generating GPU kernels
* Near-native performance for GPU kernels while using a syntax similar to Python or MATLAB
* Easy frontend API to many popular CUDA libraries
* Easy frontend API to CPU libraries (NVPL, FFTW, OpenBLAS, BLIS) on ARM and x86
* Header-only without need for compilation
* Single tensor type used across entire API
* Intuitive error messages

To start building tests, examples, and benchmarks, visit the :ref:`building` guide. This page also lists the 
requirements for each type of build.

If you're new to MatX and want to jump right in, we recommend starting with the :ref:`quickstart` guide. This
guide walks through the MatX concepts and primary API functions to give you the tools to make your first 
application using MatX. 

If you prefer learning by looking directly at complete examples, take a look in the ``examples`` directory. Some 
examples have full walk-throughs in the :ref:`examples` section of the documentation.

Lastly, an :ref:`api` guide is available for every function in MatX. 

.. note::
    We strive to make sure that every function in the MatX API has at least one example and one unit test. All example
    code in the API section is cross-referenced from the unit tests. This means that the examples in this documentation
    should always be correct and unit tests would catch any errors before they're introduced or modified in the docs.

License
-------
MatX is released under the BSD-3-Clause license. Several external packages are used optionally and may have different
license requirements. Please see the :ref:`building` guide for a list of external requirements.


Table of Contents
^^^^^^^^^^^^^^^^^
.. toctree::
    :maxdepth: 2
    :caption: MatX Documentation

    quickstart
    build.rst
    external.rst
    basics/index.rst
    api/index.rst
    examples/index.rst
    executor_compatibility
    developer_guide/index.rst
    version
