.. _quickstart:

Quick Start
===========

This guide walks through a quick introduction to MatX to get familiar with the types and basic functionality. For more extensive documentation, please
look at the following sources:

1) Example and API documentation
2) Example code in the examples/ directory
3) Jupyter notebook tutorials in the docs_input/notebooks/ directory

Adding MatX to your project
---------------------------
MatX is a single header file to include in your project called ``matx.h``. Simply add the matx/include directory to your compiler's
include search path, and add ``#include "matx.h"``. All core MatX functions are in a top-level ``matx`` namespace, while more specific functions have
a nested namespace. For example, the visualization pieces of MatX are under ``matx::viz``.

Tensor Views
------------
The most common data type in MatX is the tensor (tensor_t). The tensor is used for both viewing and managing any 
underlying GPU or host memory. While dynamically-typed languages like Python or MATLAB will implicitly allocate and manage data for the user, 
MatX requires either a one-time explicit memory allocation, or various ways of providing your own buffer. This gives more control over the lifetime 
of the data, and allows reusing memory regions for different operations.

A tensor is created using the following syntax:

.. code-block:: cpp

    auto t = make_tensor<float>({10, 20});

In this example we request a floating point tensor with rank 2 (2D array). The constructor arguments specify the shape of the tensor (10x20), 
or the size of each dimension. The number of elements in the list determines the rank of the tensor. MatX supports any arbitrary rank tensor, so the 
dimensions can be as long as you wish.

If the shape of the tensor is known at compile time, a static tensor can be created for a performance improvement when accessing elements of the
tensor:

.. code-block:: cpp

    auto t = make_static_tensor<float, 10, 20>();

Note that for a static tensor the shape is moved to the template parameters instead of function arguments.

.. note::
   While static tensors are preferred over dynamic tensors, they currently have limitations when calling certain library functions. 
   These limitations may be removed over time, but in cases where they don't work a dynamic tensor will work.

After calling the make function, MatX will allocate CUDA managed memory large enough to accommodate the specified tensor size. Users can also
pass their own pointers in a different for of the ``make_`` family of functions to allow for more control over buffer types and ownership
semantics.

With our view ``t`` created above, we now have managed memory allocated sufficiently large to hold our values, but at this point the data
in the tensor is undefined. To set individual values in a view, we can use ``operator()``:

.. code-block:: cpp

    t(2,2) = 5.5;
    t(4,6) = -10f;

The same operator can be used to get values:

.. code-block:: cpp

    float f = t(2,2) // f is now 5.5

``operator()`` takes as many parameters as the rank of the tensor:

.. code-block:: cpp

    auto f0 = t0();  // t0 is a rank-0 tensor (scalar)
    auto f1 = t1(5); // t1 is a rank-1 tensor (scalar)

Tensors can also be initialized using initializer list syntax using the ``SetVals`` function:

.. code-block:: cpp

    auto myT = make_tensor<float>({3, 3});
    myT.SetVals({{1,2,3}, {4,5,6}, {7,8,9}});

In other languages it's very common to initialize a tensor with a set of values on creation (ones, zeros, ranges). This will be covered later 
in the tutorial when we discuss operators, and it should become clear why we initialize this way.

.. note::
   For more information about creating tensors, including advanced usage, see the :ref:`creating` documentation

Getting shapes and sizes
------------------------
The dimensions of the tensor are stored internally in a type named tensorShape_t. This tensor shape contains the rank and dimensions of the
tensor view, but does not contain any information about type or storage. The shape can be retrieved using the ``Shape`` call:

.. code-block:: cpp

    auto shape = t.Shape();

``Shape()`` is similar to NumPy's ``shape`` attribute.

The number of dimensions in a tensor can be retrieved using the ``Rank()`` member. Since the rank is known at compile time, this function
uses the ``constexpr`` modifier:

.. code-block:: cpp

    auto r = t.Rank();

The size of each individual dimension can be fetched using ``Size()``:

.. code-block:: cpp

    auto t1size = t1.Size(0); // Size of vector t1
    auto t2rows = t2.Size(0); // Rows in t2
    auto t2cols = t2.Size(1); // Cols in t2

Slicing and dicing
------------------
As the name implies, ``t`` is a view into a region of memory. When the initial view is created and memory is allocated, the tensor view is
of the entire 10x20 contiguous block of memory. Often we don't want to see the entire block of memory, but only want to view a subset of the
underlying data. To do this, we use the ``slice`` operator:

.. code-block:: cpp

    auto tCube  = slice(t, {3, 5}, {6, 8});                      // Cube of t using rows 3-5 and cols 5-7
    auto tRectS = slice(t, {0, 0}, {matxEnd, matxEnd}, {2, 2});  // Rectangle with stride of 2 in both dimensions
    auto tCol   = slice<1>(t, {0, 4}, {matxEnd, matxDropDim});   // Create a 1D tensor with only column 5
    auto tRow   = slice<1>(t, {4, 0}, {matxDropDim, matxEnd});   // Create a 1D tensor with only row 5
    
``slice`` returns a new view of the tensor using start, stop, and optional stride parameters. Since views are simply
light-weight views into memory, none of these variants modify the data; they return an object with new parameters describing
how the data is viewed. The resulting variables can be used exactly as the original view above:

.. code-block:: cpp

    auto cubeRows = tCube.Size(0); // 3
    auto cubeCols = tCube.Size(1); // 3
    auto colSize  = tCol.Size(0);  // 10 since the original tensor had 10 rows
    auto rowSize  = tRow.Size(0);  // 20 since the original tensor had 20 columns

All view functions can be used on any type of existing view:

.. code-block:: cpp

    auto tCubeP  = permute(slice(t, {3, 5}, {6, 8}), {1, 0});

The above code takes the same cube as before, but permutes the cube view by swapping the two dimensions. 

``slice`` is not limited to only tensors; it can be used on any operator as input:

.. code-block:: cpp

    slice(eye(t.Shape()), {3, 5}, {6, 8});

Permuting
---------
Permuting a tensor is done using the ``permute`` function:

.. code-block:: cpp

    auto t = make_tensor<float>({10, 20});
    auto tp = permute(t, {1,0});

``tp`` is now a view into ``t`` where the rows and columns are swapped (transpose). ``permute`` is not limited to matrices, though:

.. code-block:: cpp

    auto t4 = make_tensor<float>({10, 20, 5, 2});
    auto tp4 = permute(t, {1,3,2,0});

``t4p`` is now a permuted view of the original 4D tensor, but with the dimensions swapped as ordered in the initializer list. Just like
with ``slice``, ``permute`` works on operators as well.

Note that since no data is moved, permuting a tensor can be detrimental to performance, depending on the context. Permuting usually
changes the strides of dimensions such that the memory access patterns are no longer optimal, and accessing the permuted view
continuously can be very slow. If a permuted view will be accessed repeatedly, it's recommended to copy the permuted view into
a new tensor so that the new layout is contiguous. Using the variables from above:

.. code-block:: cpp

    auto t4pc = make_tensor<float>(tp4.Shape());
    copy(t4pc, t4p);

``t4pc`` will now contain the permuted data, but in contiguous memory. Copying a tensor (or operator) can also be done by the assignment 
operator:

.. code-block:: cpp

    (t4pc = t4p).run();


Reshaping
---------
Ultimately memory is always laid out linearly regardless of how we choose to view it. We can take advantage of this property by allowing
a reshaped view of an existing view. This is commonly done when we want to take a tensor of one rank and view the data as if it were
a tensor of a different rank. The product of dimensions in one rank must equal the product of dimensions in the other rank. For example,
to take a 1D tensor of size 16 and reshape into a 2D tensor of shape 4x4::

    auto t1 = make_tensor<float>({16});
    auto t2 = t1.View({4,4});

``t2`` is now a view into the same memory as ``t1``, but viewed as a different rank. Any modifications to one tensor will be seen in the
other since no data was copied.

Increasing dimensionality
-------------------------
Sometimes it's useful to increase the rank of an existing view to match the dimensions of another tensor. For example, to add a vector onto
all rows in a matrix, you can clone the tensor to a higher rank to match the other tensor:

.. code-block:: cpp

    auto t1 = make_tensor<int>({16});
    auto t2 = make_tensor<float>({16, 16});
    // ... Initialize tensors

    auto t1c = clone<2>(t1, {16, matxKeepDim});

``t1c`` is now a new tensor view where each row is a replica of the tensor ``t1``. Again, this is just a view and no data was modified or
allocated, so modifying a row/column in either of these tensors will affect the other. 

The keyword ``matxKeepDim`` tells MatX which dimensions should be kept from the original tensor and where it should be in the new tensor.
In this example we used it in the columns place of the shape, but we also could have used ``{matxKeepDim, 16}`` and we would have a 2D
view where all columns of ``t1c`` matches ``t1``.

Note in some cases MatX's *broadcasting* feature can be used instead of ``clone``. This allows an implicit expansion of ranks during an 
element-wise operation. For example, adding a 4D tensor to a 1D tensor will work as long as the outer dimension of the 4D tensor matches
that of the 1D tensor. Broadcasting is covered in the documentation. ``clone`` is much more powerful since it gives more control over which 
dimensions are cloned instead of assuming the outer dimensions.

Creating a view from an existing pointer
----------------------------------------
While using tensor views with CUDA managed memory is very convenient, there are situations where managed memory is not ideal. Integrating
MatX into an existing codebase, or wanting more control over the memory copies are both times when using standard CUDA memory allocations
is a better option. All constructors in the tensor_t class also allow a manually-allocated pointer to be passed in. MatX will not
attempt to allocate or free any memory when this constructor is used, and it is up to the caller to manage the memory lifecycle:

.. code-block:: cpp

    float *my_device_ptr;  // Assume my_device_ptr is allocated somewhere
    auto t2 = make_tensor<float>(my_device_ptr, {20,100});
    t2(1,1) = 5; // Error! Don't do this!

In the example above, ``t2`` is a new view pointing to the existing device-allocated memory. Unlike with managed memory, ``operator()``
cannot be used on ``t2`` from the host side or the code may crash.

Operator expressions
--------------------
Tensors aren't much use by themselves if all we can do is view them in various ways. MatX provides two main ways to perform computations on
tensor views: *operator expressions* and *executors*.

Operator expressions provide a way to use algebraic expressions using tensor views and operators to generate an element-wise GPU kernel at compile-time. 
For example:

.. code-block:: cpp

    auto a = make_tensor<float>({10, 20});
    auto b = make_tensor<float>({10, 20});
    auto c = make_tensor<float>({10, 20});
    (c = a + b).run();

Ignoring that the data is unitialized, the first three lines simply create three 2D tensors with the same dimensions, while the last line runs an
operator for the equation c = a + b. In MatX terminology, an operator is a type that creates a CUDA kernel at compile-time to perform the 
element-wise operation c = a + b. The = operator is used as a deferred assignment operator expressions to avoid ambiguity with the regular assignment
operator ``=``. The ``run`` method takes an optional stream parameter, and executes the operation in the CUDA stream specified. Operators can use 
expressions of any length, and normal precedence rules apply. 

Tensor views can be mixed with scalars and operator functions:

.. code-block:: cpp

    auto op = (c = (a*a) + b / 2.0 + abs(a));

This expression squares each element in ``a``, divides each element in ``b`` by 2, adds the result to ``a``, and finally adds the resulting
tensor to the absolute value of every element in ``a``. The result of the computation will be stored in the tensor view ``c``. 
Again, the entire expression is generated at compile time and a kernel is stored in the variable ``op``, but the kernel is not launched on the device. 
To launch the operator in a CUDA stream, we use the ``run`` function:

.. code-block:: cpp

    op.run(stream);

``run`` can be thought of as a way to launch the operator/kernel into a CUDA stream, similar to the traditional triple angle bracket notation (<<<>>>). 
In MatX terminology, this is called an executor since it causes work to be executed on the device. It's often not necessary to store the operator at 
all if the execution is immediate, the two lines above can be combined:

.. code-block:: cpp

    (c = (a*a) + b / 2.0 + abs(a)).run(stream);

Sometimes the data we are using in an expression can be generated on-the-fly rather than coming from memory. Window functions, diagonal matrices, and
the identity matrix are all examples of this. MatX provides "generators" that can be used inside of expressions to generate data:

.. code-block:: cpp

    (c = (a*a) + ones(a.Shape())).run(stream);

The example above uses the ``ones`` generator to create a tensor with only the value ``1`` matching the shape of a (10x20). ``ones`` simply returns the
value ``1`` any time an element of it is requested, and no data is ever loaded from memory.

Implicit in the ``run`` call above is a CUDA executor type. As a beta feature, MatX also supports executing code on the host using a different executor.
To run the same code on the host, a ``HostExecutor`` can be passed into ``run``:

.. code-block:: cpp

    (c = (a*a) + ones(a.Shape())).run(HostExecutor{});

Instead of a CUDA stream, we pass an executor to ``run`` that instructs MatX to execute the code on the host instead of the device using a single CPU thread.
Unlike CUDA calls, host executors are synchronous, and the line above will block until finished executing.


A quick note about assignment
-----------------------------
MatX heavily relies on a deferred or lazy execution model where expressions are not executed at the time of assignment. This allows the library to 
closely match the programming model of the GPU so that there are no surprises as to when code is executed. To facilitate the asynchronous model, 
MatX overloads the assignment operator (=) to indicate a deferred execution. The deferred assignment can be executed using the ``run()`` method on 
the expression. A statement as simple as the following:

.. code-block:: cpp

    (A = B).run()

should be viewed as a deferred assignment of tensor B into tensor A (deep copy) that executes on the device when ``run()`` happens. The result of the
lazy assignment expression can also be assigned into a temporary variable:

.. code-block:: cpp

    auto op = (A = B + C);

In the code above, the ``=`` on the right side indicates lazy assignment, while the ``=`` on the left side executes the copy constructor on the new
variable ``op``. The pattern above is expected to be infrequently used since expressions are typically executed on the same line as the definition, 
but sometimes it's useful for debugging purposes to look at the type of the expression. More complex expressions follow the same rules:

.. code-block:: cpp

    IFELSE(A > 5, B = A, C = B).run()

Remember that since the assignment operator is deferred in both cases above, none of these assignments will happen until ``A > 5`` is executed on the 
device, at which point only *one* of these assignments will occur. 


Initialization of operators and generators
##########################################

As mentioned above, it's common in high-level languages to initialize a tensor/array with a known set of values. For example, generating a range of linearly-
spaced values, all ones, or a diagonal matrix. These are all operations that do not need to be generated and stored in memory before using since they are 
all generated from a formula. MatX calls these types of operators a *generator*, indicating that they generate data without storage. 

Similar to high-level languages, generators can store their values in existing tensors like so:

.. code-block:: cpp

    auto t1 = make_tensor<float>({100});
    (t1 = linspace_x(t1.Shape(), 1.0f, 100.0f)).run();

Similar to the ``set`` calls above, instead of an algebraic equation we are storing the output of generator ``linspace_x`` into the tensor ``t1``.
``linspace_x`` takes 3 parameters: the shape of the tensor (in this case we match t1), the start value, and the stop value. Since there are 100 elements
in our tensor, it will generate a sequence of 1.0, 2.0, 3.0, etc, and store it in ``t1``.

Why not just make a shorthand version of ``linspace_x`` that stores directly in a tensor? The reason is that generators can be used as part of a larger 
expression and are not limited to simply assigning to a tensor. Expanding on our last example:

.. code-block:: cpp

    auto t1 = make_tensor<float>({100});
    (t1 = ones<float>(t1.Shape()) + linspace_x(t1.Shape(), 1.0f, 100.0f) * 5.0).run();   
    
Instead of setting ``t1`` to a range, we multiply the range by 5.0, and add that range to a vector of ones using the ``ones`` generator. Without any
intermediate storage, we combined two generators, a multiply, and an add operator into a single kernel.

Transforms
----------
As mentioned above, the ``run`` function takes an executor describing where to launch the work. In the examples above ``run`` the operator
expressions created a single fused element-wise operation. Often the type of operation we are trying to do cannot be expressed as 
an element-wise operator and therefor can't be fused with other operations without synchronization. These classes of operators are called *transforms*. 
Transforms can be used anywhere operators are used:

.. code-block:: cpp

    (B = fft(A) * C).run(stream);

The ``fft`` transform above performs a 1D FFT on the tensor ``A``, multiplies the output by ``C``, and stores it in ``B``. Since the FFT
may require synchronizing before performing the multiply, MatX can internally create a temporary buffer for the FFT output and free it when
the expression goes out of scope.

Unless documented otherwise, transforms work on tensors of a specific size. Matrix multiplies require a 2D tensor (matrix), 1D FFTs require
a 1D tensor (vector), etc. If the dimension of the tensor is higher than the expected dimension, all higher dimensions will be batched. In the FFT 
call above, if ``A`` and ``B`` are 4D tensors, the inner 3 dimensions will launch a batched 1D FFT with no change in syntax.

As mentioned above, the same tensor views can be used in operator expressions before or after transforms:

.. code-block:: cpp

    (a = b + 2).run(stream);
    (c = matmul(a, d)).run(stream);

Or fused in a single line:

.. code-block:: cpp

    (c = matmul(b + 2, d)).run(stream);

The code above executes a kernel to store the result of ``b + 2`` into ``a``, then subsequently performs the matrix multiply ``C = A * B``. Since
the operator and matrix multiply are launched in the same CUDA stream, they will be executed serially.

Common reduction executors are also available, such as ``sum()``, ``mean()``, ``max()``, etc:

.. code-block:: cpp

    auto t4 = make_tensor<float>({100, 100, 100, 100});
    auto t0 = make_tensor<float>();
    (t0 = sum(t4)).run();

The above code performs an optimized sum reduction of ``t4`` into ``t0``. Currently reduction type exectors *can* take operators as an input. Please
see the documentation for a list of which ones are compatible.

For more information about operation fusion, see :ref:`fusion`.

Random numbers
--------------
MatX can generate random numbers using the cuRAND library as the backend. Random number generation consumes memory on the device, so the construction
is slightly different than other types above:

.. code-block:: cpp

    auto t2 = make_tensor<float>({100, 50});
    auto randOp = random<float>(t.Shape(), NORMAL);

The code above creates a 100x50 2D tensor, followed by a random operator that produces normally-distributed numbers with the same shape as ``t2``.

Using the random operator above uses the same assignment as with any operator, and when the values are fetched on the device a new random number
will be generated for each element.

.. code-block:: cpp

    (t2 = randOp*5 + randOp).run(stream);

In the example above ``randOp`` is accessed twice. On each access a new random number is generated.

That's it!
----------
This quick start guide was intended to give a very brief introduction to the concepts behind MatX, and how these concepts apply to the code. There's a lot
more to explore in MatX and far more functions than could be listed here. For more examples we recommend browsing through the examples to see how to perform 
real tasks using MatX, and the API guide to see an exhaustive list of functions and operators.


