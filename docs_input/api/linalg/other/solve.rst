.. _solve_func:

solve
=====

Solves the system of equations AX=Y, where X is the unknown.

.. versionadded:: 0.9.1

.. doxygenfunction:: solve(const OpA &A, const OpB &B)

Currently only supported for sparse matrix A, please see :ref:`sparse_tensor_api`.
