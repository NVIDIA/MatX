.. _zeros_func:

zeros
=====

Generate an operator of zeros

zeros() has both a shapeless and a shaped version. The shapeless version is preferred
in contexts where the shape can be deduced by the expression, thus simplifying the code.
If the shape cannot be deducded, the explicit shape version is used to specify the shape
directly.

.. doxygenfunction:: matx::zeros()
.. doxygenfunction:: matx::zeros(ShapeType &&s)
.. doxygenfunction:: matx::zeros(const index_t (&s)[RANK])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin zeros-gen-test-1
   :end-before: example-end zeros-gen-test-1
   :dedent:

