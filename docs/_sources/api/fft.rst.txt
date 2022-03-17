FFT
###
The API below provides transformations for Fast Fourier Transforms (FFTs) of both 1D and 2D types with batching.

.. doxygenfunction:: fft(OutputTensor &o, const InputTensor &i, index_t fft_size = 0, cudaStream_t stream = 0)
.. doxygenfunction:: ifft(OutputTensor &o, const InputTensor &i, index_t fft_size = 0, cudaStream_t stream = 0)
.. doxygenfunction:: fft2(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: ifft2(OutputTensor &o, const InputTensor &i, cudaStream_t stream = 0)
.. doxygenfunction:: dct(OutputTensor &out, const InputTensor &in, const cudaStream_t stream = 0)
