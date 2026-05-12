.. _svd_func:

svd
###

Perform a singular value decomposition (SVD). 

.. versionadded:: 0.6.0

.. note::
   ``svd`` is a multi-output solver API and is not currently supported by CUDA JIT fusion or cuSolverDx in MatX.
   Use a normal non-JIT executor path for SVD; ``CUDAJITExecutor`` rejects this operator.

.. doxygenfunction:: svd

Enums
~~~~~

The following enums are used for configuring the behavior of SVD operations.

.. doxygenenum:: SVDMode
.. doxygenenum:: SVDHostAlgo


Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/SVD.cu
   :language: cpp
   :start-after: example-begin svd-test-1
   :end-before: example-end svd-test-1
   :dedent:
