.. _ifft2_func:

ifft2
#####

Perform a 2D inverse FFT. Batching is supported for any tensor with a rank higher than 2.


.. doxygenfunction:: ifft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD)
.. doxygenfunction:: ifft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD)
  
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