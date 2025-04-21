.. _interp_func:

interp1
=======

Piecewise linear or nearest-neighbor interpolation.

.. doxygenfunction:: interp1(const OpX &x, const OpV &v, const OpXQ &xq)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/interp_test.cu
   :language: cpp
   :start-after: example-begin interp-test-1
   :end-before: example-end interp-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/interp_test.cu
   :language: cpp
   :start-after: example-begin interp-test-2
   :end-before: example-end interp-test-2
   :dedent:
