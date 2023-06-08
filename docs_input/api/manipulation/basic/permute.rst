.. _permute_func:

permute
#######

Permute the dimensions of an operator

.. doxygenfunction:: permute(detail::tensor_impl_t<T, Rank> &out, const detail::tensor_impl_t<T, Rank> &in, const std::initializer_list<uint32_t> &dims, const cudaStream_t stream)
.. doxygenfunction:: permute(const T &op, const int32_t (&dims)[T::Rank()])
.. doxygenfunction:: permute(const T &op, const std::array<int32_t, T::Rank()> &dims)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin permute-test-1
   :end-before: example-end permute-test-1
   :dedent:
