.. _ifft_func:

ifft
####

Perform a 1D inverse FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: ifft(OutputTensor o, const InputTensor i, uint64_t fft_size = 0, cudaStream_t stream = 0)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft-1
  :end-before: example-end ifft-1
  :dedent:

.. doxygenfunction:: ifft(OutputTensor out, const InputTensor in, const int (&axis)[1], uint64_t fft_size = 0, cudaStream_t stream = 0)

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft-2
  :end-before: example-end ifft-2
  :dedent: