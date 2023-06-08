.. _hanning_func:

hanning
=======

Generate a Hanning window

.. doxygenfunction:: matx::hanning(ShapeType &&s)
.. doxygenfunction:: matx::hanning(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin hanning-gen-test-1
   :end-before: example-end hanning-gen-test-1
   :dedent:

