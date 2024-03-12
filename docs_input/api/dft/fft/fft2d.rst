.. _fft2_func:

fft2
####

Perform a 2D FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: fft2(OpA &&a, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: fft2(OpA &&a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD)  

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft2-1
  :end-before: example-end fft2-1
  :dedent:

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin fft2-2
  :end-before: example-end fft2-2
  :dedent:  