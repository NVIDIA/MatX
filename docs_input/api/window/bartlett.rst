.. _bartlett_func:

bartlett
========

Generate a Bartlett window

.. doxygenfunction:: matx::bartlett(ShapeType &&s)
.. doxygenfunction:: matx::bartlett(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin bartlett-gen-test-1
   :end-before: example-end bartlett-gen-test-1
   :dedent:

