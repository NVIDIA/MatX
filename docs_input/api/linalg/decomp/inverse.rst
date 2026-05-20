.. _inv_func:

inv
===

Matrix inverse
--------------

Compute the inverse of a square matrix.

.. note::
   This function is currently not supported with host-based executors (CPU)

.. note::
   CUDA JIT kernel fusion is supported by cuSolverDx if ``-DMATX_EN_MATHDX=ON`` is enabled. The current JIT path
   supports rank 2 through 4 square matrices with ``float``, ``double``, ``complex<float>``, and
   ``complex<double>`` values. Unsupported ranks, shapes, or data types should use a normal executor and will be
   rejected by ``CUDAJITExecutor``.

.. note::
   The default inverse uses LU/GETRF-style factorization for general square matrices. For Hermitian
   positive-definite inputs, ``inv<MAT_INVERSE_ALGO_POSV>(A)`` enables a cuSolverDx POSV JIT path that solves
   ``A * X = I`` directly and can fuse with other compatible JIT operators such as cuBLASDx matmul. POSV is
   currently a MathDx/CUDAJITExecutor-only inverse algorithm, and callers must ensure the input satisfies the
   positive-definite contract.

.. versionadded:: 0.6.0

.. doxygenfunction:: inv(const OpA &a)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Inverse.cu
   :language: cpp
   :start-after: example-begin inv-test-1
   :end-before: example-end inv-test-1
   :dedent:
