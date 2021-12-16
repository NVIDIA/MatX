FFT
###
The API below provides transformations for Fast Fourier Transforms (FFTs) of both 1D and 2D types with batching.

Cached API
----------
.. doxygenfunction:: fft(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: ifft(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: fft2(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: ifft2(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: dct(OutputTensor &out, const InputTensor &in, const cudaStream_t stream = 0)

Non-Cached API
--------------
.. doxygenclass:: matx::matxFFTPlan1D_t
    :members:
.. doxygenclass:: matx::matxFFTPlan2D_t
    :members: