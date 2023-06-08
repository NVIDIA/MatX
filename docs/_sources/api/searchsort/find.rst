.. _find_func:

find
====

Find values in an operator based on a selection operator and returns the values. Users can define their own custom comparator
function or choose from a built-in set of common comparators. The values meeting the selection criteria
are returned in `a_out`, while the number of elements found are in `num_found`. It's important that `a_out`
is sized large enough to store all elements found or the behavior is undefined.

.. doxygenfunction:: find(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, cudaExecutor exec = 0)
.. doxygenfunction:: find(OutputTensor &a_out, CountTensor &num_found, const InputOperator &a, SelectType sel, SingleThreadHostExecutor exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin find-test-1
   :end-before: example-end find-test-1
   :dedent:

