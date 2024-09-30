.. _select_func:

select
=======

Selects value from a tensor based on a 1D index mapping. The 1D index mapping works for any rank tensor.
Usually the mapping is provided by `find_idx`, but any source with the same mapping will work.

.. doxygenfunction:: select(const T &t, IdxType idx)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin select-test-1
   :end-before: example-end select-test-1
   :dedent:

