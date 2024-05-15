.. _mean_func:

mean
====

Compute the mean of the reduction dimensions

.. doxygenfunction:: mean(const InType &in, const int (&dims)[D])
.. doxygenfunction:: mean(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin mean-test-1
   :end-before: example-end mean-test-1
   :dedent:

