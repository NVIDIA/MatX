.. _polyval_func:

polyval
=======

Evaluate a polynomial given an input sequence and coefficients

.. doxygenfunction:: polyval(const Op &op, const Coeffs &coeffs)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/polyval_test.cu
   :language: cpp
   :start-after: example-begin polyval-test-1
   :end-before: example-end polyval-test-1
   :dedent:

