.. _argmax_func:

argmax
======

Returns both the maximum values and the indices of the maximum values across the input operator

.. doxygenfunction:: argmax(const InType &in, const int (&dims)[D])
.. doxygenfunction:: argmax(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argmax-test-1
   :end-before: example-end argmax-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin argmax-test-2
   :end-before: example-end argmax-test-2
   :dedent:
