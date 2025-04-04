.. _clone_func:

clone
=====

Clone one or more dimensions of an operator to a higher rank

.. doxygenfunction:: clone(const Op &t, const index_t (&shape)[Rank])
.. doxygenfunction:: clone(const Op &t, const cuda::std::array<index_t, Rank> &shape)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/clone_test.cu
   :language: cpp
   :start-after: example-begin clone-test-1
   :end-before: example-end clone-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/clone_test.cu
   :language: cpp
   :start-after: example-begin clone-test-2
   :end-before: example-end clone-test-2
   :dedent:
