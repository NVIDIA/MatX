.. _rmax_func:

rmax
====

Reduces the input by the maximum values across the specified axes. Note the name `rmax` is used as to note
collide with the C++ standard library or other libraries with `max` defined.

.. doxygenfunction:: rmax(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
.. doxygenfunction:: rmax(OutType dest, const InType &in, cudaExecutor exec = 0)
.. doxygenfunction:: rmax(OutType dest, const InType &in, SingleThreadHostExecutor exec)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin rmax-test-1
   :end-before: example-end rmax-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin rmax-test-2
   :end-before: example-end rmax-test-2
   :dedent:
