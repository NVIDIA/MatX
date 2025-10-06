.. _ifft2_func:

ifft2
#####

Perform a 2D inverse FFT. Batching is supported for any tensor with a rank higher than 2.


.. doxygenfunction:: ifft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: ifft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD)

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
  :start-after: example-begin ifft2-1
  :end-before: example-end ifft2-1
  :dedent:

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft2-2
  :end-before: example-end ifft2-2
  :dedent:  