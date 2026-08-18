.. _reduce_func:

reduce
======

Reduces the input  using a generic reduction operator and optionally store the indices of the reduction

For CUDA tensor views whose final two dimensions are a transpose of contiguous
storage, a reduction of the innermost dimension uses coalesced tiled loads.
Other permutations and strided layouts retain the general CUB iterator path.

.. versionadded:: 0.6.0

.. doxygenfunction:: reduce(const InType &in, ReduceOp op, bool init = true)
.. doxygenfunction:: reduce(const InType &in, const int (&dims)[D], ReduceOp op, bool init = true)
