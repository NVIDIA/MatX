.. _interp_func:

interp
=======

Piecewise linear or nearest-neighbor interpolation.

.. doxygenfunction:: interp(const OpX &x, const OpV &v, const OpXQ &xq)
.. doxygenfunction:: interp(const OpX &x, const OpV &v, const OpXQ &xq, const InterpMethod method)

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
