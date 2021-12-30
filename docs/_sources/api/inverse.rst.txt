Matrix Inverse
##############

The following functions provide an interface for performing a matrix inverse using the LU decomposition
method. The inverse API is currently using cuBLAS as a backend for the LU decomposition.

Cached API
----------
.. doxygenfunction:: inv(TensorTypeAInv &a_inv, const TensorTypeA &a, cudaStream_t stream = 0)

