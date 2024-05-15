.. _linspace_func:

linspace
========

Return a range of linearly-spaced numbers using first and last value. The step size is
determined by the shape.

.. doxygenfunction:: matx::linspace(ShapeType &&s, T first, T last)
.. doxygenfunction:: matx::linspace(const index_t (&s)[RANK], T first, T last)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin linspace-gen-test-1
   :end-before: example-end linspace-gen-test-1
   :dedent:

