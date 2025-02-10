.. _mul_func:

Multiply (*)
============

Element-wise multiplication operator

.. doxygenfunction:: operator*(Op t, Op t2)
.. doxygenfunction:: mul(Op t, Op t2)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/operator_func_test.cu
   :language: cpp
   :start-after: example-begin mul-test-1
   :end-before: example-end mul-test-1
   :dedent:

