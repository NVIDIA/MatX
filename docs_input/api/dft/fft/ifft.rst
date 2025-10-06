.. _ifft_func:

ifft
####

Perform a 1D inverse FFT. Batching is supported for any tensor with a rank higher than 1.


.. doxygenfunction:: ifft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: ifft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)

FFT Normalization
~~~~~~~~~~~~~~~~~

The ``norm`` parameter specifies how the FFT and inverse FFT are scaled:

- ``FFTNorm::BACKWARD``: FFT is unscaled, inverse FFT is scaled by 1/N (default)
- ``FFTNorm::FORWARD``: FFT is scaled by 1/N, inverse FFT is unscaled
- ``FFTNorm::ORTHO``: Both FFT and inverse FFT are scaled by 1/sqrt(N)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft-1
  :end-before: example-end ifft-1
  :dedent:


.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft-2
  :end-before: example-end ifft-2
  :dedent: