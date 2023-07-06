.. _eye_func:

eye
===

Generate an identity tensor

``eye()`` has both a shaped and a shape-less form. The shaped form takes a single argument specifying
the shape of the operator. This is useful when the operator is used in contexts where it must have a shape.
For example:

.. code-block:: cpp

    auto kroneye  = kron(eye({4, 4}));

Without a shape, the ``kron`` operator would not be able to generate a Kronecker product, and will result
in a compiler error. 

Shapeless is useful when the size is already known by another operator:

.. code-block:: cpp

    auto t2 = make_tensor<float>({5, 5});
    (t2 = eye()).run();

In the case above the lazy assignment of ``t2`` is done at runtime and will only request elements 0:5,0:5
since the number of elements fetched is dictated by the size of ``t2``.

.. doxygenfunction:: matx::eye(ShapeType &&s)
.. doxygenfunction:: matx::eye(const index_t (&s)[RANK])
.. doxygenfunction:: matx::eye()

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin eye-gen-test-1
   :end-before: example-end eye-gen-test-1
   :dedent:

