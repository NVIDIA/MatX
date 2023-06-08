.. _trace_func:

trace
=====

Return the trace (sum of elements on the diagonal) of a tensor

.. doxygenfunction:: trace(OutType dest, const InType &in, Executor &&exec)
.. doxygenfunction:: trace(OutType dest, const InType &in, int stream)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin trace-test-1
   :end-before: example-end trace-test-1
   :dedent:

