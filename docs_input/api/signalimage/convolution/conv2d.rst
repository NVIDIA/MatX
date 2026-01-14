.. _conv2d_func:

conv2d
######

2D convolution

.. versionadded:: 0.6.0

.. doxygenfunction:: conv2d(const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode)
.. doxygenfunction:: conv2d(const In1Type &i1, const In2Type &i2, const int32_t (&axis)[2], matxConvCorrMode_t mode)   

Convolution/Correlation Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``mode`` parameter specifies how the output size is determined:

- ``MATX_C_MODE_FULL``: Keep all elements including ramp-up/down (output size = N + M - 1)
- ``MATX_C_MODE_SAME``: Keep only elements where entire filter was present (output size = max(N, M))
- ``MATX_C_MODE_VALID``: Keep only elements with full overlap (output size = max(N, M) - min(N, M) + 1)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv2d-test-1
   :end-before: example-end conv2d-test-1
   :dedent:
