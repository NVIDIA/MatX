.. _reduce_func:

reduce
======

Reduces the input  using a generic reduction operator and optionally store the indices of the reduction

.. doxygenfunction:: reduce(const InType &in, ReduceOp op, bool init = true)
.. doxygenfunction:: reduce(const InType &in, const int (&dims)[D], ReduceOp op, bool init = true)
