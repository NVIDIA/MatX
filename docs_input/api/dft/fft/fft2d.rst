.. _fft2_func:

fft2
####

Perform a 2D FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: fft2(OutputTensor o, const InputTensor i, cudaStream_t stream = 0)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft2-1
  :end-before: example-end fft2-1
  :dedent:

.. doxygenfunction:: fft2(OutputTensor out, const InputTensor in, const int (&axis)[2], cudaStream_t stream = 0)

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft2-2
  :end-before: example-end fft2-2
  :dedent:  