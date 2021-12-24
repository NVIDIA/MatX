Welcome to MatX's documentation!
======================================================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

This site contains all documentation for MatX, including API docs, tutorials, 
and example code. 

MatX is a modern C++ library for numerical computing on NVIDIA GPUs and CPUs. 
Near-native performance can be achieved while using a simple syntax common in 
higher-level languages such as Python or MATLAB. To achieve this performance,
MatX uses a combination of compile-time optimizations and existing CUDA libraries
to hide complexity from the developer.

A single data type (tensor_t) is used by both algebraic expressions and backend CUDA
libraries. Tensors can be any rank, and virtually any data type. Types that are optimized
for specific hardware will use the accelerated features if present. For example, tensor
cores will be used when fp16 or bf16 inputs are used in a GEMM operation.

To get started with MatX, please read the quickstart_ guide, and for a complete API
overview visit the api_ index.


Table of Contents
^^^^^^^^^^^^^^^^^
.. toctree::
    :maxdepth: 2

    build
    quickstart
    creation
    api/index    
    matlabpython
    examples/index
    limitations