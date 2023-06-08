.. _var_func:

var
###

Compute the variance of a tensor. `ddof` can be used optionally to control the bias term in the denominator

.. doxygenfunction:: var(OutType dest, const InType &in, int ddof = 1, int stream = 0)
.. doxygenfunction:: var(OutType dest, const InType &in, Executor &&exec, int ddof = 1)
.. doxygenfunction:: var(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec, int ddof = 1)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin var-test-1
   :end-before: example-end var-test-1
   :dedent:
