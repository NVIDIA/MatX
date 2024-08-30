.. _einsum:

einsum
######

.. versionadded:: 0.2.3

MatX provides an ``einsum`` function similar to the one found in NumPy. ``einsum`` allows
a brief syntax to express many different operations in an optimized manner. A non-exhaustive list
of ``einsum`` operations are:

* Tensor contractions
* Matrix multiplies (GEMMs)
* Inner products
* Transposes
* Reductions
* Trace

While many of these operations are possible using other methods in MatX, ``einsum`` typically has a 
shorter syntax, and is sometimes more optimized than a direct version of the operation. 

.. note::
   Using einsum() requires a minimum of cuTENSOR 1.7.0 and cuTensorNet 23.03.0.20. These are downloaded
   automatically as part of CMake, but for offline environments these versions are needed.

As of now, MatX only supports a limited set of ``einsum`` operations that would be supported in
the NumPy version. Specifically only tensor contractions, inner products, and GEMMs are supported 
and tested at this time. MatX also does not support broadcast '...' notation and has no plans to. While
the broadcast notation is useful for high-rank tensors, it doesn't add any new features and
isn't compatible in all expressions inside NumPy. We feel listing out the dimensions makes the 
syntax more clear without giving up any features. Since ``einsum`` requires an output tensor parameter, 
only *explicit* mode is supported using the ``->`` operator. This allows type and size checking on the 
output tensor at the cost of extra verbosity.

For tensor contractions, MatX uses cuTENSOR and cuTensorNet as the optimized backend libraries. Since
neither of these libraries are included with CUDA, and not all users need ``einsum`` functionality, ``einsum``
is an opt-in feature when configuring MatX. To add support, add the following CMake line:

.. code-block:: shell

    -DMATX_EN_CUTENSOR=ON

Both cuTENSOR and cuTensorNet can have their location specified using ``cutensor_DIR`` and ``cutensornet_DIR``, 
respectively. If these are not specified, CMake will attempt to download both libraries from the internet. ``einsum`` 
is inside the ``cutensor`` namespace in MatX to indicate that it's an optional feature. 

To perform a tensor contraction of two 3D tensors across a single dimension:

.. code-block:: cpp

    auto a = make_tensor<float>({3,4,5});
    auto b = make_tensor<float>({4,3,2});
    auto c = make_tensor<float>({5,2});
    cutensor::einsum(c, "ijk,jil->kl", 0, a, b);

The letters in the ``subscripts`` argument are names given to each dimension. The letters are arbitrary, but the 
dimensions being contracted must have matching letters. In this case, we're contracting along the ``i`` and ``j``
dimensions of both tensor ``a`` and ``b``, resulting in an output tensor with dimensions ``k x l``. The tensor ``c``
must match the output dimensions of ``k x l``, which are the third dimension of both ``a`` and ``b`` (5 x 2).

Like other features in MatX, ``einsum`` can take an in artbirary number of tensors with arbitrary ranks. Each tensor's
dimensions are separated by ``,`` in the subscript list, and the actual tensors are enumerated at the end of the function.

The first time ``einsum`` runs a contraction of a certain signature it can take a long time to complete. this is
because cuTensorNet and cuTensor perform optimization heuristics for future contractions. The penalty is
only paid on the first call of input tensors with that signature, and subsequent calls will only perform the 
contraction step.

.. note::
   einsum's permute capability is significantly faster than the permute operator and should be preferred when possible.

.. note::
   This function is currently not supported with host-based executors (CPU)

API
---

.. doxygenfunction:: einsum

Examples
--------

Tensor Contractions
~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-contraction-1
   :end-before: example-end einsum-contraction-1
   :dedent:

Dot Product
~~~~~~~~~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-dot-1
   :end-before: example-end einsum-dot-1
   :dedent:

GEMM (Generalized Matrix Multiply)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-gemm-1
   :end-before: example-end einsum-gemm-1
   :dedent:   

.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-gemm-2
   :end-before: example-end einsum-gemm-2
   :dedent:

Permute
~~~~~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-permute-1
   :end-before: example-end einsum-permute-1
   :dedent:

Sum
~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-sum-1
   :end-before: example-end einsum-sum-1
   :dedent:   

Trace
~~~~~
.. literalinclude:: ../../../../test/00_tensor/EinsumTests.cu
   :language: cpp
   :start-after: example-begin einsum-trace-1
   :end-before: example-end einsum-trace-1
   :dedent:      