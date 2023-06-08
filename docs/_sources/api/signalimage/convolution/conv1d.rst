.. _conv1d_func:

conv1d
######

1D convolution

.. doxygenfunction:: conv1d(OutputType o, const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode, cudaStream_t stream = 0)

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

.. doxygenfunction:: conv1d(OutputType o, const In1Type &i1, const In2Type &i2, const int32_t (&axis)[1], matxConvCorrMode_t mode, cudaStream_t stream = 0)   

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin conv1d-test-3
   :end-before: example-end conv1d-test-3
   :dedent: