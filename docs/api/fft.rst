FFT
###
The API below provides transformations for Fast Fourier Transforms (FFTs) of both 1D and 2D types with batching.

Cached API
----------
.. doxygenfunction:: fft(tensor_t<T1, RANK> &o, const tensor_t<T2, RANK> &i, cudaStream_t stream = 0)
.. doxygenfunction:: ifft(tensor_t<T1, RANK> &o, const tensor_t<T2, RANK> &i, cudaStream_t stream = 0)
.. doxygenfunction:: fft2(tensor_t<T1, RANK> &o, const tensor_t<T2, RANK> &i, cudaStream_t stream = 0)
.. doxygenfunction:: ifft2(tensor_t<T1, RANK> &o, const tensor_t<T2, RANK> &i, cudaStream_t stream = 0)
.. doxygenfunction:: dct(tensor_t<T, RANK> &out, tensor_t<T, RANK> &in, const cudaStream_t stream = 0)

Non-Cached API
--------------
.. doxygenclass:: matx::matxFFTPlan1D_t
    :members:
.. doxygenclass:: matx::matxFFTPlan2D_t
    :members: