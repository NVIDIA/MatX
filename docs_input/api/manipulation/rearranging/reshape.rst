.. _reshape_func:

reshape
=======

Reshape an operator by giving it new sizes. The total size of the reshaped operator must match
the original size.

.. doxygenfunction:: reshape(const T &op, ShapeType &&s)
.. doxygenfunction:: reshape( const T &op, const index_t (&sizes)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/reshape_test.cu
   :language: cpp
   :start-after: example-begin reshape-test-1
   :end-before: example-end reshape-test-1
   :dedent:

