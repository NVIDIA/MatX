.. _sort_func:

sort
####

Sort elements of a tensor in either ascending or descending order

.. doxygenfunction:: sort(const InputOperator &a, const SortDirection_t dir)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin sort-test-1
   :end-before: example-end sort-test-1
   :dedent:


.. literalinclude:: ../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin sort-test-2
   :end-before: example-end sort-test-2
   :dedent:



