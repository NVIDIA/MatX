.. _median_func:

median
======

Compute the median of the reduction dimensions

.. doxygenfunction:: median(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: median(OutType dest, const InType &in, SingleThreadHostExecutor exec)
.. doxygenfunction:: median(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin median-test-1
   :end-before: example-end median-test-1
   :dedent:

