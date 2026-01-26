.. _argsort_func:

argsort
#######

Compute the indices that would sort the elements of a tensor in either ascending or descending order

.. versionadded:: 0.6.0

.. doxygenfunction:: argsort(const InputOperator &a, const SortDirection_t dir)

Sort Direction
~~~~~~~~~~~~~~

The ``dir`` parameter specifies the sort direction:

- ``SORT_DIR_ASC``: Sort in ascending order (smallest to largest)
- ``SORT_DIR_DESC``: Sort in descending order (largest to smallest)

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



