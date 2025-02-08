.. _abs2_func:

abs2
====

Squared absolute value. For complex numbers, this is the squared
complex magnitude, or real(t)\ :sup:`2` + imag(t)\ :sup:`2`. For real numbers,
this is equivalent to the squared value, or t\ :sup:`2`.

.. doxygenfunction:: abs2(Op t) 

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/abs2_test.cu
   :language: cpp
   :start-after: example-begin abs2-test-1
   :end-before: example-end abs2-test-1
   :dedent:

