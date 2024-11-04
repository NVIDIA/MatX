.. _argminmax_func:

argminmax
=========

Returns the minimum values, minimum value indices, maximum values, and maximum value indices across the input operator

.. doxygenfunction:: argminmax(const InType &in, const int (&dims)[D])
.. doxygenfunction:: argminmax(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argminmax-test-1
   :end-before: example-end argminmax-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argminmax-test-2
   :end-before: example-end argminmax-test-2
   :dedent:
