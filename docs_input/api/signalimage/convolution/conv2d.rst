.. _conv2d_func:

conv2d
######

2D convolution

.. doxygenfunction:: conv2d(const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode)
.. doxygenfunction:: conv2d(const In1Type &i1, const In2Type &i2, const int32_t (&axis)[2], matxConvCorrMode_t mode)   

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv2d-test-1
   :end-before: example-end conv2d-test-1
   :dedent:
