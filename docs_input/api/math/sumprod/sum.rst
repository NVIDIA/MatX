.. _sum_func:

sum
===

Reduces the input by the sum of values across the specified axes.

.. doxygenfunction:: sum(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: sum(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: sum(OutType dest, const InType &in,  SingleThreadHostExecutor exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin sum-test-1
   :end-before: example-end sum-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin sum-test-2
   :end-before: example-end sum-test-2
   :dedent:
