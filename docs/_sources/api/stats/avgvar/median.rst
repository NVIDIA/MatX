.. _median_func:

median
======

Compute the median of the reduction dimensions

.. doxygenfunction:: median(const InType &in, const int (&dims)[D])
.. doxygenfunction:: median(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin median-test-1
   :end-before: example-end median-test-1
   :dedent:

