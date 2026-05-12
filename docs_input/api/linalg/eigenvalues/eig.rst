.. _eig_func:

eig
###

Perform an eigenvalue decomposition for Hermitian or real symmetric matrices.

.. versionadded:: 0.6.0

.. note::
   ``eig`` is a multi-output solver API and is not currently supported by CUDA JIT fusion or cuSolverDx in MatX.
   Use a normal non-JIT executor path for eigenvalue decomposition; ``CUDAJITExecutor`` rejects this operator.

.. doxygenfunction:: eig

Enums
~~~~~

The following enums are used for configuring the behavior of Eig operations.

.. doxygenenum:: EigenMode


Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Eigen.cu
   :language: cpp
   :start-after: example-begin eig-test-1
   :end-before: example-end eig-test-1
   :dedent:
