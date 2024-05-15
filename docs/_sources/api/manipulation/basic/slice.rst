.. _slice_func:

slice
#####

Slice an operator with new start and end points, and optionally new strides. Slice can also
be used to drop ranks, for operations such as selecting a single row. 

.. doxygenfunction:: slice(const OpType opIn, const index_t (&starts)[OpType::Rank()], const index_t (&ends)[OpType::Rank()])
.. doxygenfunction:: slice(const OpType op, const index_t (&starts)[OpType::Rank()], const index_t (&ends)[OpType::Rank()], const index_t (&strides)[OpType::Rank()])

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin slice-test-1
   :end-before: example-end slice-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin slice-test-2
   :end-before: example-end slice-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin slice-test-2
   :end-before: example-end slice-test-2
   :dedent:

