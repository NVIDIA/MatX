.. _logspace_func:

logspace
========

Return a range of logarithmically-spaced numbers using first and last value. The step size is
determined by the shape.

.. doxygenfunction:: matx::logspace(ShapeType &&s, T first, T last)
.. doxygenfunction:: matx::logspace(const index_t (&s)[RANK], T first, T last)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin logspace-gen-test-1
   :end-before: example-end logspace-gen-test-1
   :dedent:

