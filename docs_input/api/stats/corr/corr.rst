.. _corr_func:

corr
####

Cross-correlation of two inputs

.. versionadded:: 0.6.0

.. doxygenfunction:: corr(const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode, matxConvCorrMethod_t method)
.. doxygenfunction:: corr(const In1Type &i1, const In2Type &i2, const int32_t (&axis)[1], matxConvCorrMode_t mode, matxConvCorrMethod_t method)

Convolution/Correlation Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``mode`` parameter specifies how the output size is determined:

- ``MATX_C_MODE_FULL``: Keep all elements including ramp-up/down (output size = N + M - 1)
- ``MATX_C_MODE_SAME``: Keep only elements where entire filter was present (output size = max(N, M))
- ``MATX_C_MODE_VALID``: Keep only elements with full overlap (output size = max(N, M) - min(N, M) + 1)

Convolution/Correlation Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``method`` parameter specifies the algorithm to use:

- ``MATX_C_METHOD_DIRECT``: Direct convolution using sliding window approach
- ``MATX_C_METHOD_FFT``: FFT-based convolution using the convolution theorem (may be faster for large inputs)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin corr-test-1
   :end-before: example-end corr-test-1
   :dedent:
