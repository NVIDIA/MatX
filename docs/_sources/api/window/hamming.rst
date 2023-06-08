.. _hamming_func:

hamming
=======

Generate a Hamming window

.. doxygenfunction:: matx::hamming(ShapeType &&s)
.. doxygenfunction:: matx::hamming(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin hamming-gen-test-1
   :end-before: example-end hamming-gen-test-1
   :dedent:

