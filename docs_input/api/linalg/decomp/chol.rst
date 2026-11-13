.. _chol_func:

chol
####

Perform a Cholesky factorization.

.. note::
  The input matrix must be symmetric positive-definite

.. note::
  CUDA JIT kernel fusion is supported by cuSolverDx if ``-DMATX_EN_MATHDX=ON`` is enabled. The current JIT path
  supports rank 2 through 4 square matrices with ``float``, ``double``, ``complex<float>``, and
  ``complex<double>`` values. Unsupported ranks, shapes, or data types should use a normal executor and will be
  rejected by ``CUDAJITExecutor``.

.. versionadded:: 0.6.0

.. doxygenfunction:: chol(const OpA &a, SolverFillMode uplo = SolverFillMode::UPPER)
.. doxygenfunction:: chol(const OpA &a, const SolverStatus &status, SolverFillMode uplo = SolverFillMode::UPPER)

.. note::
   The overload that takes a :ref:`solver status <solver_status_func>` writes each matrix's potrf ``info`` value
   into the status instead of synchronizing the stream to check it. A positive value i means the leading minor of order i is not positive definite. It is not supported with
   ``CUDAJITExecutor``.

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
