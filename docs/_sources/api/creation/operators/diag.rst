.. _diag_func:

diag
====

`diag` comes in two forms: a generator and an operator. The generator version is used to generate a diagonal
tensor with a given value, while the operator pulls diagonal elements from a tensor.

Operator
________
.. doxygenfunction:: matx::diag(T1 t)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-op-test-1
   :end-before: example-end diag-op-test-1
   :dedent:

The generator form of ``diag()`` has both a shaped and a shape-less form. The shaped form takes a single argument specifying
the shape of the operator. This is useful when the operator is used in contexts where it must have a shape.
For example:

.. code-block:: cpp

    auto krondiag  = kron(diag({4, 4}, 5));

Without a shape, the ``kron`` operator would not be able to generate a Kronecker product, and will result
in a compiler error.

Shapeless is useful when the size is already known by another operator:

.. code-block:: cpp

    auto t2 = make_tensor<float>({5, 5});
    (t2 = diag(5)).run();

In the case above the lazy assignment of ``t2`` is done at runtime and will only request elements 0:5,0:5
since the number of elements fetched is dictated by the size of ``t2``.

Generator
_________

.. doxygenfunction:: matx::diag(const index_t (&s)[RANK], T val)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-gen-test-1
   :end-before: example-end diag-gen-test-1
   :dedent:
