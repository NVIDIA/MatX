.. _all_func:

all
===

Returns a truth value if all values in the reduction converts to a boolean "true"

.. doxygenfunction:: all(const InType &in, const int (&dims)[D])
.. doxygenfunction:: all(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin all-test-1
   :end-before: example-end all-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin all-test-2
   :end-before: example-end all-test-2
   :dedent:
