.. _cross_func:

cross
=====

Cross product of two operators with last dimension 2 or 3.

Inputs `A` and `B` may be higher rank than 1, in which case batching will occur
on all dimensions besides the last dimension.

.. doxygenfunction:: cross(const OpA &A, const OpB &B)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/cross_test.cu
   :language: cpp
   :start-after: example-begin cross-test-1
   :end-before: example-end cross-test-1
   :dedent:
