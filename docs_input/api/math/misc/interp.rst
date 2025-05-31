.. _interp_func:

interp1
=======

Piecewise interpolation with various methods (linear, nearest, next, previous, spline).

.. doxygenfunction:: interp1(const OpX &x, const OpV &v, const OpXQ &xq, InterpMethod method)
.. doxygenfunction:: interp1(const OpX &x, const OpV &v, const OpXQ &xq, const int (&axis)[1], InterpMethod method)

Interpolation Methods
~~~~~~~~~~~~~~~~~~~~~

.. doxygenenum:: matx::InterpMethod

Examples
~~~~~~~~

Linear Interpolation (default):

.. literalinclude:: ../../../../test/00_operators/interp_test.cu
   :language: cpp
   :start-after: example-begin interp-test-1
   :end-before: example-end interp-test-1
   :dedent:

Nearest Neighbor Interpolation:

.. literalinclude:: ../../../../test/00_operators/interp_test.cu
   :language: cpp
   :start-after: example-begin interp-test-2
   :end-before: example-end interp-test-2
   :dedent:
