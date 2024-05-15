.. _ifft_func:

ifft
####

Perform a 1D inverse FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: ifft(OpA &&a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: ifft(OpA &&a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)

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