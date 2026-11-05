.. _slice_func:

slice
#####

Slice an operator with new start and end points, and optionally new strides. Slice can also
be used to drop ranks, for operations such as selecting a single row. Negative indices can be used 
to indicate starting at the end and going backward.

When slicing along any given tensor dimension, the start index is treated as inclusive, and the end index as exclusive.
Start, end, and stride arrays always refer to the input dimensions, including dimensions removed with
``matxDropDim``. For example, for an input whose values are ``10 * row + column``,
``slice<1>(input + 0, {1, 2}, {matxDropDim, matxEnd})`` selects row 1 starting at column 2.
For a 3 by 5 input this produces ``12, 13, 14``; adding strides ``{1, 2}`` produces ``12, 14``.

With the CUDA JIT executor (``MATX_EN_JIT=ON``), expression slice starts and strides are compiled
into the generated kernel. Different resolved starts or strides therefore require separate JIT
kernel specializations, even when the output shapes match.

.. versionadded:: 0.3.0

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

