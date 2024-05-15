.. _chol_func:

chol
####

Perform a Cholesky factorization and saves the result in either the upper or lower triangle of the output. 

.. note::
  The input matrix must be positive semidefinite

.. doxygenfunction:: chol

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Cholesky.cu
   :language: cpp
   :start-after: example-begin chol-test-1
   :end-before: example-end chol-test-1
   :dedent:
