.. _permute_func:

permute
#######

Permute the dimensions of an operator

.. versionadded:: 0.3.0

.. doxygenfunction:: permute(const T &op, const int32_t (&dims)[T::Rank()])
.. doxygenfunction:: permute(const T &op, const cuda::std::array<int32_t, T::Rank()> &dims)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/permute_test.cu
   :language: cpp
   :start-after: example-begin permute-test-1
   :end-before: example-end permute-test-1
   :dedent:

Adjacent permutations of expressions are composed into a single permutation. Inverse permutations therefore restore the original mapping,
including for in-place element-wise operations with unsafe alias detection enabled. A non-identity combined permutation that reads the destination
remains unsafe. Each permutation's axes are validated before composition.
