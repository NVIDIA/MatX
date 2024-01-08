.. _outer_func:

outer
#####

Outer product of two vectors

Inputs `A` and `B` may be higher rank than 1, in which case batching will occur
on all other dimensions.

.. doxygenfunction:: outer

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin outer-test-1
   :end-before: example-end outer-test-1
   :dedent:
