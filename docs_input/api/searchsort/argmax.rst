.. _argmax_func:

argmax
======

Returns both the maximum values and the indices of the maximum values across the input operator

.. doxygenfunction:: argmax(OutType dest, const TensorIndexType &idest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: argmax(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: argmax(OutType dest, TensorIndexType &idest, const InType &in, SingleThreadHostExecutor exec) 

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
