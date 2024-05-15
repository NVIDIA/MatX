.. _find_idx_func:

find_idx
========

Finds values in an operator based on a selection operator and returns the indices where the value is. Users can define their own custom comparator
function or choose from a built-in set of common comparators. The indices meeting the selection criteria
are returned in `a_out`, while the number of elements found are in `num_found`. It's important that `a_out`
is sized large enough to store all elements found or the behavior is undefined.

.. doxygenfunction:: find_idx(const OpA &a, SelectType sel)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin find_idx-test-1
   :end-before: example-end find_idx-test-1
   :dedent:

