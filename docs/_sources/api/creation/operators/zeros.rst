.. _zeros_func:

zeros
=====

Generate an operator of zeros

.. doxygenfunction:: matx::zeros(ShapeType &&s)
.. doxygenfunction:: matx::zeros(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin zeros-gen-test-1
   :end-before: example-end zeros-gen-test-1
   :dedent:

