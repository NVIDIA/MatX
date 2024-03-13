.. _min_func:

min
===

Reduces the input by the minimum values across the specified axes.

.. doxygenfunction:: min(const InType &in, const int (&dims)[D])
.. doxygenfunction:: min(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin min-test-1
   :end-before: example-end min-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin min-test-2
   :end-before: example-end min-test-2
   :dedent:
