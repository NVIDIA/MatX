.. _cumsum_func:

cumsum
======

Compute the cumulative sum of the reduction dimensions

.. doxygenfunction:: cumsum(OutputTensor &a_out, const InputOperator &a, cudaExecutor exec = 0)
.. doxygenfunction:: cumsum(OutputTensor &a_out, const InputOperator &a, SingleThreadHostExecutor exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin cumsum-test-1
   :end-before: example-end cumsum-test-1
   :dedent:

