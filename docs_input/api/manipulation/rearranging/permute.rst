.. _permute_func:

permute
#######

Permute the dimensions of an operator

.. doxygenfunction:: permute(const T &op, const int32_t (&dims)[T::Rank()])
.. doxygenfunction:: permute(const T &op, const cuda::std::array<int32_t, T::Rank()> &dims)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/permute_test.cu
   :language: cpp
   :start-after: example-begin permute-test-1
   :end-before: example-end permute-test-1
   :dedent:
