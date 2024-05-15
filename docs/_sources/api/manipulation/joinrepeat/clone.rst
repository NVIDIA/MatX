.. _clone_func:

clone
=====

Clone one or more dimensions of an operator to a higher rank

.. doxygenfunction:: clone(Op t, const index_t (&shape)[Rank])
.. doxygenfunction:: clone(Op t, const std::array<index_t, Rank> &shape)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin clone-test-1
   :end-before: example-end clone-test-1
   :dedent:

