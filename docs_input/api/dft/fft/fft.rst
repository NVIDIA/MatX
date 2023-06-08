.. _fft_func:

fft
###

Perform a 1D FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: fft(OutputTensor o, const InputTensor i, uint64_t fft_size = 0, cudaStream_t stream = 0)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft-1
  :end-before: example-end fft-1
  :dedent:

.. doxygenfunction:: fft(OutputTensor out, const InputTensor in, const int32_t (&axis)[1], uint64_t fft_size = 0, cudaStream_t stream = 0)

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