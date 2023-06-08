.. _lu_func:

lu
##

Perform an LU factorization. The input and output tensors may be the same tensor, in which case the
input is overwritten.

.. doxygenfunction:: lu

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/LU.cu
   :language: cpp
   :start-after: example-begin lu-test-1
   :end-before: example-end lu-test-1
   :dedent:
