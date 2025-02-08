.. _transpose_func:

transpose
#########

Transpose an operator into a tensor. This is equivalent to permuting the dimensions of
an operator in reverse order. Using `transpose()` potentially allows for higher performance
than calling `permute()` since it's not lazily evaluated and can use an optimized implementation.

.. doxygenfunction:: transpose(const T &op)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/transpose_test.cu
   :language: cpp
   :start-after: example-begin transpose-test-1
   :end-before: example-end transpose-test-1
   :dedent:
