.. _argmin_func:

argmin
======

Returns both the minimum values and the indices of the minimum values across the input operator

.. doxygenfunction:: argmin(OutType dest, TensorIndexType &idest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: argmin(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: argmin(OutType dest, TensorIndexType &idest, const InType &in, SingleThreadHostExecutor exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argmin-test-1
   :end-before: example-end argmin-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argmin-test-2
   :end-before: example-end argmin-test-2
   :dedent:
