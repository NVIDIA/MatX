.. _shift_func:

shift
=====

Shift an operator by a given amount either positive or negative along one dimension

.. doxygenfunction:: shift(const OpT &op, ShiftOpT s)
.. doxygenfunction:: shift(const OpT &op, ShiftT s, ShiftsT... shifts)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/shift_test.cu
   :language: cpp
   :start-after: example-begin shift-test-1
   :end-before: example-end shift-test-1
   :dedent:

