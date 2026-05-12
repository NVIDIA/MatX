.. _lu_func:

lu
##

Perform an LU factorization.

.. versionadded:: 0.6.0

.. note::
   The ``mtie`` assignment form of ``lu`` uses the normal non-JIT solver path. CUDA JIT fusion is available through
   lazy projection members such as ``lu(A).LU`` and ``lu(A).Piv`` when ``-DMATX_EN_MATHDX=ON`` is enabled and the
   runtime shape/type is supported by cuSolverDx. Projection JIT currently supports ranks 2 through 4 and
   ``float``, ``double``, ``complex<float>``, and ``complex<double>`` inputs.

.. doxygenfunction:: lu

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/LU.cu
   :language: cpp
   :start-after: example-begin lu-test-1
   :end-before: example-end lu-test-1
   :dedent:
