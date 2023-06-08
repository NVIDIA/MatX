.. _stdd_func:

stdd
####

Compute the standard deviation of a tensor. The name `stdd` is used to avoid confliction with the use of the C++ standard library

.. doxygenfunction:: matx::stdd(OutType dest, const InType &in, Executor &&exec)
.. doxygenfunction:: matx::stdd(OutType dest, const InType &in, int stream = 0)
.. doxygenfunction:: matx::stdd(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin stdd-test-1
   :end-before: example-end stdd-test-1
   :dedent:
