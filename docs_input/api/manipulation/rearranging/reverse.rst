.. _reverse_func:

reverse
=======

Reverse the values of an operator along a single dimension

.. cpp:function:: template <int DIM, typename Op> reverse(const Op &t)
.. cpp:function:: template <int DIM1, int DIM2, int... DIMS, typename Op> reverse(const Op &t)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/reverse_test.cu
   :language: cpp
   :start-after: example-begin reverse-test-1
   :end-before: example-end reverse-test-1
   :dedent:

