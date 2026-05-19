.. _eig_func:

eig
###

Perform an eigenvalue decomposition for Hermitian or real symmetric matrices.

.. versionadded:: 0.6.0

.. note::
   The ``mtie`` assignment form of ``eig`` uses the normal non-JIT solver path. CUDA JIT fusion is available through
   lazy projection members such as ``eig(A).Vectors`` and ``eig(A).Values`` when ``-DMATX_EN_MATHDX=ON`` is enabled
   and the runtime shape/type is supported by cuSolverDx. Projection JIT currently supports rank 2 through 4 square
   Hermitian or real symmetric matrices with ``float``, ``double``, ``complex<float>``, and ``complex<double>``
   inputs. Expressions that only reference ``Values`` use the cuSolverDx no-vectors job.

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
