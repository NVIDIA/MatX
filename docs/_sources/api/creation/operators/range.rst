.. _range_func:

range
=====

Return a range of numbers using a start and step value. The total length is based on the shape

.. doxygenfunction:: matx::range(ShapeType &&s, T first, T step)
.. doxygenfunction:: matx::range(const index_t (&s)[RANK], T first, T step)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin range-gen-test-1
   :end-before: example-end range-gen-test-1
   :dedent:

