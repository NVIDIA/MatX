.. _slice_func:

slice
#####

Slice an operator with new start and end points, and optionally new strides. Slice can also
be used to drop ranks, for operations such as selecting a single row. Negative indices can be used 
to indicate starting at the end and going backward.

When slicing along any given tensor dimension, the start index is treated as inclusive, and the end index as exclusive.

.. doxygenfunction:: slice(const OpType &op, const index_t (&starts)[OpType::Rank()], const index_t (&ends)[OpType::Rank()], const index_t (&strides)[OpType::Rank()])
.. doxygenfunction:: slice( const OpType &op, const index_t (&starts)[OpType::Rank()], const index_t (&ends)[OpType::Rank()]) 

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/slice_test.cu
   :language: cpp
   :start-after: example-begin slice-test-1
   :end-before: example-end slice-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/slice_stride_test.cu
   :language: cpp
   :start-after: example-begin slice-test-2
   :end-before: example-end slice-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/slice_and_reduce_test.cu
   :language: cpp
   :start-after: example-begin slice-test-3
   :end-before: example-end slice-test-3
   :dedent:

.. literalinclude:: ../../../../test/00_operators/slice_test.cu
   :language: cpp
   :start-after: example-begin slice-test-4
   :end-before: example-end slice-test-4
   :dedent:   

