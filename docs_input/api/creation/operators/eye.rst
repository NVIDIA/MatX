.. _eye_func:

eye
===

Generate an identity tensor

.. doxygenfunction:: matx::eye(ShapeType &&s)
.. doxygenfunction:: matx::eye(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin eye-gen-test-1
   :end-before: example-end eye-gen-test-1
   :dedent:

