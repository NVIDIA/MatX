.. _ones_func:

ones
====

Generate an operator of ones

.. doxygenfunction:: matx::ones(ShapeType &&s)
.. doxygenfunction:: matx::ones(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin ones-gen-test-1
   :end-before: example-end ones-gen-test-1
   :dedent:

