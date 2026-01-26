.. _any_func:

any
===

Returns a truth value if any value in the reduction converts to a boolean "true"

.. versionadded:: 0.6.0

.. doxygenfunction:: any(const InType &in, const int (&dims)[D])
.. doxygenfunction:: any(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin any-test-1
   :end-before: example-end any-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin any-test-2
   :end-before: example-end any-test-2
   :dedent:
