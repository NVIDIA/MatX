.. _sum_func:

sum
===

Reduces the input by the sum of values across the specified axes.

On CUDA, reducing the innermost dimension of a tensor view produced by
transposing the final two dimensions uses a tiled fast path. This keeps reads
coalesced without materializing the transpose. Other permutations and strided
views continue to use the general reduction path.

.. versionadded:: 0.6.0

.. doxygenfunction:: sum(const InType &in, const int (&dims)[D])
.. doxygenfunction:: sum(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin sum-test-1
   :end-before: example-end sum-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin sum-test-2
   :end-before: example-end sum-test-2
   :dedent:
