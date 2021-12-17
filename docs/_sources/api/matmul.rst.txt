Matrix Multiply (GEMM)
######################

The API below provides transformation functions for Generic Matrix Multiplies (GEMMs) for complex and real values, and batching support
for tensors of range 3 and 4.

Cached API
----------
.. doxygenfunction:: matmul

Non-Cached API
--------------
.. doxygenclass:: matx::matxMatMulHandle_t
    :members:
.. doxygenenum:: matx::MatXMatMulProvider_t
