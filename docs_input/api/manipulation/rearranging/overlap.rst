.. _overlap_func:

overlap
#######

Create an overlapping view an of input operator giving a higher-rank view of the input

For example, the following 1D tensor [1 2 3 4 5] could be cloned into a 2d tensor with a
window size of 2 and overlap of 1, resulting in::

  [1 2
   2 3
   3 4
   4 5]

Currently this only works on 1D tensors going to 2D, but may be expanded
for higher dimensions in the future. Note that if the window size does not
divide evenly into the existing column dimension, the view may chop off the
end of the data to make the tensor rectangular.

.. note::
    Only 1D input operators are accepted at this time

.. doxygenfunction:: overlap( const OpType &op, const index_t (&windows)[N], const index_t (&strides)[N])
.. doxygenfunction:: overlap( const OpType &op, const cuda::std::array<index_t, N> &windows, const cuda::std::array<index_t, N> &strides)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/overlap_test.cu
   :language: cpp
   :start-after: example-begin overlap-test-1
   :end-before: example-end overlap-test-1
   :dedent:
