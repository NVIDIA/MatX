.. _stdd_func:

stdd
####

Compute the standard deviation of a tensor. The name `stdd` is used to avoid confliction with the use of the C++ standard library

.. doxygenfunction:: stdd(const InType &in, const int (&dims)[D], int ddof = 1)
.. doxygenfunction:: stdd(const InType &in, int ddof = 1)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin stdd-test-1
   :end-before: example-end stdd-test-1
   :dedent:
