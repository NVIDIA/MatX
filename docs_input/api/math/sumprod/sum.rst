.. _sum_func:

sum
===

Reduces the input by the sum of values across the specified axes.

.. doxygenfunction:: sum(const InType &in, const int (&dims)[D])
.. doxygenfunction:: sum(const InType &in)

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
