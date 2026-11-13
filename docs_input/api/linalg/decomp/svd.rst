.. _svd_func:

svd
###

Perform a singular value decomposition (SVD). 

.. versionadded:: 0.6.0

.. note::
   ``svd`` is not currently supported by CUDA JIT fusion in MatX. The runtime ``libcusolverdx`` code-generation API
   used by MatX does not expose an SVD routine, so use a normal non-JIT executor path for SVD. Lazy projection
   members such as ``svd(A).U``, ``svd(A).S``, and ``svd(A).VT`` are available on normal executors and can be used
   in expressions with other operators.

.. doxygenfunction:: svd(const OpA &a, const SVDMode jobz = SVDMode::ALL, const SVDHostAlgo algo = SVDHostAlgo::DC)
.. doxygenfunction:: svd(const OpA &a, const SolverStatus &status, const SVDMode jobz = SVDMode::ALL, const SVDHostAlgo algo = SVDHostAlgo::DC)

.. note::
   The overload that takes a :ref:`solver status <solver_status_func>` writes each matrix's SVD ``info`` value
   into the status instead of synchronizing the stream to check it. A positive value means the algorithm did not converge. It is not supported with
   ``CUDAJITExecutor``.

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

Projection Examples
~~~~~~~~~~~~~~~~~~~

The ``U``, ``S``, and ``VT`` projections can be composed with other operators to reconstruct the input matrix.
This example is included from the ``ProjectionAPI`` unit test.

.. literalinclude:: ../../../../test/00_solver/SVD.cu
   :language: cpp
   :start-after: example-begin svd-projection-test-1
   :end-before: example-end svd-projection-test-1
   :dedent:
