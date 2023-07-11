.. _concepts:

Concepts
========

MatX uses numerous concepts that are given names to help understand how the library works. This document
compiles a list of the terms and definitions that will be useful when reading other documents and code.

Operator
--------

The most important concept in MatX is the `Operator`. An `Operator` is any type that adheres to the `Operator interface`.
Specifically, any operator must implement the following functions:

* Size
* Shape
* Stride
* Rank
* operator() (For rvalues at a minimum, and optionally lvalues)

These functions provide a minimum set of functionality that can be used on both the host and device. 

The API definition above is simple, but very powerful. Any class/type that implements these functions 
can be used anywhere an operator is accepted. This includes: arithmetic expressions, transforms,
assignments, etc. `Operators` are a superset containing other types, including generators and tensors.

As an example:

.. code-block:: cpp

    // Create a tensor. "t" is an operator
    auto t    = make_tensor<float>({10);

    // Create a sin operator that operates on "t" and name it "op"
    auto op   = sin(t);

    // Create a Hamming window generator and assign it to the variable "win"
    auto win  = hamming({10});

    // Launch a kernel where "win" is copied into "t"
    (t = win).run();

The first three lines have a single operator created. In order, they create a tensor, operator, and generator, respectively.
As mentioned above, ``t``, ``sin(t)``, and ``hamming({10})`` all have the operator interface above implemented.

The last line more nuanced in where the operator is used. In ``t = win`` the operator is the assignment operator (``=``), 
which in this case produces a lazily-evaluated assignment  of ``win`` into ``t``. ``=`` implements the entire operator
interface above and is used as an operator in the assignment expression.

Tensor
______

Tensors are memory-backed operators providing several initialization methods for different use cases. Tensors implement the
entire operator interface above, plus many more helper functions specific to a tensor type. Tensors can be used as both lvalues
and rvalues in every place an operator is expected.

For more information on creating tensors, see :ref:`creating`. 

Generator
_________

Generators are a type of operator that can generate values without another tensor or operator as input. For example, windowing
functions, such as a Hamming window, can generate values by only taking a length as input. Generators are efficient since they
require no memory.

Transform
---------

Some functions in MatX can only be executed on a single line without any other operators. For example, an fft is executed by:

.. code-block:: cpp

    fft(A, A);

It is currently not valid to do something like the following:

.. code-block:: cpp

    (C = B * fft(A, A)).run();

The reason this is invalid is because functions that are classified as transforms launch CUDA kernels to perform a single function,
and many times they call a CUDA library. Transforms are not operators and cannot be used in operator expressions as shown above.
Since ``fft`` is not an operator the compiler will give an error.

This behavior may change in the future or be relaxed for certain transforms.

Since some transforms rely on CUDA math library backends not all of them are available with different executors. Please see the
documentation for the individual function to check compatibility.

Executor
--------

Executors are types that describe how to execute an operator expression or transform. They are similar to C++'s execution policy, and
may even use C++ execution policies behind the scenes. Executors are designed so that the code can remain unchanged while executing 
on a variety of different targets. Currently the following executors are defined:

* cudaExecutor - Execute on a CUDA-supported device
* SingleThreadHostExecutor - Execute on a single host thread

More executor types will be added in future releases.

Shape
-----

Shape is used to describe the size of each dimension of an operator.

Stride
------

Stride is used to describe the spacing between elements in each dimension of an operator