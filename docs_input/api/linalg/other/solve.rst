.. _solve_func:

solve
=====

Solves the system of equations AX=Y, where X is the unknown.

.. versionadded:: 0.9.1

.. doxygenfunction:: solve(const OpA &A, const OpB &B)

For dense ``A``, ``A`` must have shape ``... x n x n``. ``B`` may have shape
``... x n`` for a vector right-hand side or ``... x n x nrhs`` for one or more
matrix right-hand sides. Dense batch dimensions must match exactly; they are not
broadcast. The output shape matches ``B``. Dense solve uses cuSolver/cuBLAS with
CUDA executors and LAPACK with host executors when CPU solver support is
configured.

For sparse ``A``, ``solve`` preserves the legacy sparse solve layout where
right-hand sides are stacked by row and the operation solves ``A X^T = B^T``.
Sparse direct solve support depends on the sparse format and may require cuDSS;
please see :ref:`sparse_tensor_api`.

Examples
~~~~~~~~

Dense vector right-hand side

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-1
   :end-before: example-end solve-test-1
   :dedent:

Dense matrix right-hand side

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-2
   :end-before: example-end solve-test-2
   :dedent:

In an expression

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-3
   :end-before: example-end solve-test-3
   :dedent:

Batched vector right-hand side

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-4
   :end-before: example-end solve-test-4
   :dedent:

Batched matrix right-hand side

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-5
   :end-before: example-end solve-test-5
   :dedent:

In-place right-hand side

.. literalinclude:: ../../../../test/00_solver/SolveDense.cu
   :language: cpp
   :start-after: example-begin solve-test-6
   :end-before: example-end solve-test-6
   :dedent:
