.. _chol_func:

chol
####

Perform a Cholesky factorization.

.. note::
  The input matrix must be symmetric positive-definite

.. doxygenfunction:: chol

Enums
~~~~~

The following enums are used for configuring the behavior of Cholesky operations.

.. _solverfillmode:

.. doxygenenum:: SolverFillMode

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Cholesky.cu
   :language: cpp
   :start-after: example-begin chol-test-1
   :end-before: example-end chol-test-1
   :dedent:
