.. _conv2d_func:

conv2d
######

2D convolution

.. doxygenfunction:: conv2d(OutputType o, const In1Type in1, const In2Type in2, matxConvCorrMode_t mode, cudaStream_t stream = 0)
.. doxygenfunction:: conv2d(OutputType &o, const In1Type &i1, const In2Type &i2, const int32_t (&axis)[2],  matxConvCorrMode_t mode, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv2d-test-1
   :end-before: example-end conv2d-test-1
   :dedent:
