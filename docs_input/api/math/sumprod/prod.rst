.. _prod_func:

prod
====

Reduce an input by the product of all elements in the reduction set

.. doxygenfunction:: prod(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: prod(OutType dest, const InType &in, SingleThreadHostExecutor exec)
.. doxygenfunction:: prod(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin prod-test-1
   :end-before: example-end prod-test-1
   :dedent:

