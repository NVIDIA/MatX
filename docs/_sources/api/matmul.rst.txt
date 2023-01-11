Matrix Multiply (GEMM)
######################

The API below provides transformation functions for Generic Matrix Multiplies (GEMMs) for complex and real values. Batching
is supported for any tensor with a rank higher than 2.

Cached API
----------
.. doxygenfunction:: matmul(TensorTypeC C, const TensorTypeA A, const TensorTypeB B, const int32_t (&axis)[2], cudaStream_t stream = 0, float alpha = 1.0, float beta = 0.0)
.. doxygenfunction:: matmul(TensorTypeC C, const TensorTypeA A, const TensorTypeB B, cudaStream_t stream = 0, float alpha = 1.0, float beta = 0.0)

