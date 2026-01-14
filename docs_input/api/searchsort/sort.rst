.. _sort_func:

sort
####

Sort elements of a tensor in either ascending or descending order

.. versionadded:: 0.6.0

.. doxygenfunction:: sort(const InputOperator &a, const SortDirection_t dir)

Sort Direction
~~~~~~~~~~~~~~

The ``dir`` parameter specifies the sort direction:

- ``SORT_DIR_ASC``: Sort in ascending order (smallest to largest)
- ``SORT_DIR_DESC``: Sort in descending order (largest to smallest)

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



