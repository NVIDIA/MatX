.. _mean_func:

mean
====

Compute the mean of the reduction dimensions

.. doxygenfunction:: mean(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: mean(OutType dest, const InType &in, SingleThreadHostExecutor exec)
.. doxygenfunction:: mean(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin mean-test-1
   :end-before: example-end mean-test-1
   :dedent:

