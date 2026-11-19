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
.. doxygenfunction:: inv(const OpA &a, const SolverStatus &status)

.. note::
   The overload that takes a :ref:`solver status <solver_status_func>` writes each matrix's factorization ``info``
   value into stage 0 and its inversion ``info`` value into stage 1, instead of synchronizing the stream and throwing
   for a singular matrix. The status needs at least two stages. It supports the LU algorithm only and is not
   supported with ``CUDAJITExecutor``.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Inverse.cu
   :language: cpp
   :start-after: example-begin inv-test-1
   :end-before: example-end inv-test-1
   :dedent:
