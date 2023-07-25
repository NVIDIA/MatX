.. _prod_func:

prod
====

Reduce an input by the product of all elements in the reduction set

.. doxygenfunction:: prod(const InType &in, const int (&dims)[D])
.. doxygenfunction:: prod(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin prod-test-1
   :end-before: example-end prod-test-1
   :dedent:

