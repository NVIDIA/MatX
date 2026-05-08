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
