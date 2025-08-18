.. _pad_func:

pad
===

Pad operators along a dimension. The returned operator will have the same rank as the input operator.
The sizes of the unpadded dimensions will be the same as the input operator and the padded dimension
will increase by the sum of the pre- and post-padding sizes.

The currently suported padding modes are MATX_PAD_MODE_CONSTANT and MATX_PAD_MODE_EDGE. The constant
padding mode uses the specified pad value for all padded elements. The edge padding mode uses the edge
values of the input operator for the padded elements (i.e., pre-padding will use the value of the first
element and post-padding will use the value of the last element).

.. doxygenenum:: matx::PadMode

.. doxygenfunction:: pad(const T& op, int axis, const std::array<index_t, 2>& pad_sizes, const typename T::value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT)
.. doxygenfunction:: pad(const T& op, int axis, const index_t (&pad_sizes)[2], const typename T::value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/pad_test.cu
   :language: cpp
   :start-after: example-begin pad-test-1
   :end-before: example-end pad-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/pad_test.cu
   :language: cpp
   :start-after: example-begin pad-test-2
   :end-before: example-end pad-test-2
   :dedent:
