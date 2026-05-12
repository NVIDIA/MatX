.. _lu_func:

lu
##

Perform an LU factorization.

.. versionadded:: 0.6.0

.. note::
   ``lu`` is a multi-output solver API and is not currently supported by CUDA JIT fusion or cuSolverDx in MatX.
   Use a normal non-JIT executor path for LU factorization; ``CUDAJITExecutor`` rejects this operator.

.. doxygenfunction:: lu

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/LU.cu
   :language: cpp
   :start-after: example-begin lu-test-1
   :end-before: example-end lu-test-1
   :dedent:
