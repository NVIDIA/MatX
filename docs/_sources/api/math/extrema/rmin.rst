.. _rmin_func:

rmin
====

Reduces the input by the minimum values across the specified axes. Note the name `rmin` is used as to note
collide with the C++ standard library or other libraries with `min` defined.

.. doxygenfunction:: rmin(const InType &in, const int (&dims)[D])
.. doxygenfunction:: rmin(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin rmin-test-1
   :end-before: example-end rmin-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin rmin-test-2
   :end-before: example-end rmin-test-2
   :dedent:
