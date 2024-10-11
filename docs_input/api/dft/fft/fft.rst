.. _fft_func:

fft
###

Perform a 1D FFT. Batching is supported for any tensor with a rank higher than 1.


.. doxygenfunction:: fft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: fft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-1
  :end-before: example-end fft-1
  :dedent:


.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-2
  :end-before: example-end fft-2
  :dedent:

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-3
  :end-before: example-end fft-3
  :dedent:

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-4
  :end-before: example-end fft-4
  :dedent:

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-5
  :end-before: example-end fft-5
  :dedent: