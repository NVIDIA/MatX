.. _shift_func:

shift
=====

Shift an operator by a given amount either positive or negative along one dimension.
The shift amount can be a scalar (uniform shift) or a rank-1 operator to shift each
row or column by a different amount.

.. versionadded:: 0.3.0

.. doxygenfunction:: shift(const OpT &op, ShiftOpT s)
.. doxygenfunction:: shift(const OpT &op, ShiftT s, ShiftsT... shifts)

Examples
~~~~~~~~

Shift all rows by the same amount:

.. literalinclude:: ../../../../test/00_operators/shift_test.cu
   :language: cpp
   :start-after: example-begin shift-test-1
   :end-before: example-end shift-test-1
   :dedent:

Shift each column by a different amount using a rank-1 operator:

.. literalinclude:: ../../../../test/00_operators/shift_test.cu
   :language: cpp
   :start-after: example-begin shift-test-2
   :end-before: example-end shift-test-2
   :dedent:

