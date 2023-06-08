.. _corr_func:

corr
####

Cross-correlation of two inputs

.. doxygenfunction:: corr(OutputTensor o, const In1Type i1, const In2Type i2,  matxConvCorrMode_t mode, matxConvCorrMethod_t method, cudaStream_t stream = 0)
.. doxygenfunction:: corr(OutputTensor o, const In1Type i1, const In2Type i2, const int32_t (&axis)[1], matxConvCorrMode_t mode, matxConvCorrMethod_t method, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin corr-test-1
   :end-before: example-end corr-test-1
   :dedent:
