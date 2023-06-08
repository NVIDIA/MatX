.. _blackman_func:

blackman
========

Generate a Blackman window

.. doxygenfunction:: matx::blackman(ShapeType &&s)
.. doxygenfunction:: matx::blackman(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin blackman-gen-test-1
   :end-before: example-end blackman-gen-test-1
   :dedent:

