.. _flattop_func:

flattop
=======

Generate a flattop window

.. doxygenfunction:: matx::flattop(ShapeType &&s)
.. doxygenfunction:: matx::flattop(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin flattop-gen-test-1
   :end-before: example-end flattop-gen-test-1
   :dedent:

