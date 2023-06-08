.. _any_func:

any
===

Returns a truth value if any value in the reduction converts to a boolean "true"

.. doxygenfunction:: any(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: any(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: any(OutType dest, const InType &in, SingleThreadHostExecutor exec)

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
