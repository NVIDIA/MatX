.. _ones_func:

ones
====

Generate an operator of ones

ones() has both a shapeless and a shaped version. The shapeless version is preferred
in contexts where the shape can be deduced by the expression, thus simplifying the code.
If the shape cannot be deducded, the explicit shape version is used to specify the shape
directly.

.. doxygenfunction:: matx::ones()
.. doxygenfunction:: matx::ones(ShapeType &&s)
.. doxygenfunction:: matx::ones(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin ones-gen-test-1
   :end-before: example-end ones-gen-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin ones-gen-test-2
   :end-before: example-end ones-gen-test-2
   :dedent:


