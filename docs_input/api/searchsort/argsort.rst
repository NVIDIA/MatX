.. _argsort_func:

argsort
#######

Compute the indices that would sort the elements of a tensor in either ascending or descending order

.. doxygenfunction:: argsort(const InputOperator &a, const SortDirection_t dir)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin argsort-test-1
   :end-before: example-end argsort-test-1
   :dedent:


.. literalinclude:: ../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin argsort-test-2
   :end-before: example-end argsort-test-2
   :dedent:



