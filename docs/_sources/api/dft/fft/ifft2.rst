.. _ifft2_func:

ifft2
#####

Perform a 2D inverse FFT

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: ifft2(OutputTensor o, const InputTensor i, cudaStream_t stream = 0)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft2-1
  :end-before: example-end ifft2-1
  :dedent:

.. doxygenfunction:: ifft2(OutputTensor out, const InputTensor in, const int (&axis)[2], cudaStream_t stream = 0)

.. literalinclude:: ../../../../test/00_transform/FFT.cu
  :language: cpp
  :start-after: example-begin ifft2-2
  :end-before: example-end ifft2-2
  :dedent:  