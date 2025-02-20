.. _stack_func:

stack
=====

Stack operators along a dimension. Each input must be the same rank, and the returned operator has
a rank increase of one where the new dimension reflects the stacked operators.

.. doxygenfunction:: stack

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/stack_test.cu
   :language: cpp
   :start-after: example-begin stack-test-1
   :end-before: example-end stack-test-1
   :dedent:

