.. _all_func:

all
===

Returns a truth value if all values in the reduction converts to a boolean "true"

.. doxygenfunction:: all(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: all(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: all(OutType dest, const InType &in, SingleThreadHostExecutor exec)

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
