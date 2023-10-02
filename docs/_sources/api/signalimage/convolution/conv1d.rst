.. _conv1d_func:

conv1d
######

1D convolution

Performs a convolution operation of two inputs. Three convolution modes are available: full, same, and valid. The
mode controls how much (if any) of the output is truncated to remove filter ramps. The method parameter allows
either direct or FFT-based convolution. Direct performs the typical sliding-window dot product approach, whereas
FFT uses the convolution theorem. The FFT method may be faster for large inputs, but both methods should be tested
for the target input sizes.

.. doxygenfunction:: conv1d(const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode, matxConvCorrMethod_t method)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv1d-test-1
   :end-before: example-end conv1d-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv1d-test-2
   :end-before: example-end conv1d-test-2
   :dedent:

.. doxygenfunction:: conv1d(const In1Type &i1, const In2Type &i2, const int32_t (&axis)[1], matxConvCorrMode_t mode = MATX_C_MODE_FULL, matxConvCorrMethod_t method = MATX_C_METHOD_DIRECT)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv1d-test-3
   :end-before: example-end conv1d-test-3
   :dedent: